# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare finite expert credit against native routed computation, including drops."""

import pytest
from levanter.testing.cpu_devices import run_on_cpu_devices

# These cases require the pinned runtime prepared by prepare.py.
pytestmark = pytest.mark.manual


def test_routed_matrix_credit_with_capacity_drops_on_two_devices():
    run_on_cpu_devices(
        device_count=2,
        script="""
        import equinox as eqx
        import jax
        import jax.numpy as jnp
        import numpy as np
        from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P
        from levanter.grug._moe.ep_fixed_pooled_wave_all_to_all import _moe_mlp_ep_fixed_pooled_wave_a2a_local
        from experiments.grug.noise_belay.routed_expert_credit import (
            pooled_relu2_parameter_credit, pooled_router_parameter_credit,
        )

        rng = np.random.default_rng(41)
        # Deliberately overloaded expert 0 and destination shard 0 cause both
        # receiver and sender drops, including tokens losing every assignment.
        selected_np = np.asarray([[0,1]]*6 + [[2,3]]*2 + [[0,2]]*6 + [[2,3]]*2, np.int32)
        x_np = rng.normal(size=(16,6)).astype(np.float32)
        weights_np = rng.uniform(.1, 1, size=(16,2)).astype(np.float32)
        up_np = rng.normal(size=(4,6,8)).astype(np.float32)
        down_np = rng.normal(size=(4,8,5)).astype(np.float32)
        credit_np = rng.normal(size=(16,5)).astype(np.float32)
        router_np = rng.normal(size=(6,4)).astype(np.float32)*.1
        axes = ('replica_dcn','data','expert')
        for count in (1,2):
            mesh = Mesh(np.asarray(jax.devices()[:count]).reshape(1,1,count,1),
                        (*axes,'model'), axis_types=(AxisType.Explicit,)*4)
            with jax.set_mesh(mesh):
                specs = (P(axes,None),P(axes,None),P(axes,None),P('expert',None,None),P('expert',None,None),P(axes,None))
                x, selected, weights, up, down, credit = [
                    jax.device_put(a, NamedSharding(mesh,p)) for a,p in zip(
                        (x_np,selected_np,weights_np,up_np,down_np,credit_np), specs)]
                def local_forward(x,selected,weights,up,down,credit):
                    out,drops = _moe_mlp_ep_fixed_pooled_wave_a2a_local(
                        x,selected,weights,jnp.ones(x.shape[0],jnp.bool_),up,down,
                        activation_fn=lambda y: jnp.square(jax.nn.relu(y)),
                        num_experts=4,capacity_factor=.7,token_sharding_axes=axes,
                        transport_capacity_factor=.8,num_expert_waves=1,fp8_dispatch=False,
                        expert_remat=False,report_assignment_keep=True)
                    keep = drops.assignment_keep
                    kept_sum = jnp.sum(jnp.where(keep,weights,0),axis=-1)
                    scale = jnp.where(kept_sum>0, jnp.sum(weights,axis=-1)/jnp.maximum(kept_sum,1e-20),1)
                    loss = jax.lax.psum(jnp.sum(out*scale[:,None]*credit),axes)
                    return loss,jax.lax.psum(jnp.sum(~keep),axes)
                native = jax.shard_map(local_forward,mesh=mesh,in_specs=specs,out_specs=(P(),P()),check_vma=False)
                (_, dropped), expected = jax.jit(jax.value_and_grad(native,argnums=(3,4),has_aux=True))(
                    x,selected,weights,up,down,credit)
                actual = eqx.filter_jit(pooled_relu2_parameter_credit)(
                    x,selected,weights,up,down,credit,capacity_factor=.7,
                    transport_capacity_factor=.8,relative_step=.0001)
                assert int(dropped)>0
                # At this step scale no ReLU crossing occurs in the random test;
                # the finite-cycle and derivative oracle agree to float32 precision.
                for a,e in zip(actual,expected):
                    np.testing.assert_allclose(a,e,atol=2e-4,rtol=2e-4)

                router = jax.device_put(router_np, NamedSharding(mesh,P(None,None)))
                def router_objective(router):
                    logits = jnp.einsum('td,de->te',x,router,out_sharding=P(axes,None))
                    chosen = jnp.take_along_axis(logits,selected,axis=-1)
                    mass = jnp.sqrt(jax.nn.softplus(chosen))
                    combine = 2.5*mass/(mass.sum(axis=-1,keepdims=True)+1e-9)
                    return native(x,selected,combine,up,down,credit)[0]
                expected_router = jax.jit(jax.grad(router_objective))(router)
                logits = jnp.einsum('td,de->te',x,router,out_sharding=P(axes,None))
                chosen = jnp.take_along_axis(logits,selected,axis=-1)
                actual_router = eqx.filter_jit(pooled_router_parameter_credit)(
                    x,selected,chosen,up,down,credit,x,capacity_factor=.7,
                    transport_capacity_factor=.8,sigma=.001,renorm_sum=2.5)
                np.testing.assert_allclose(actual_router,expected_router,atol=2e-4,rtol=2e-4)
        """,
    )
