# pip install eopod -qU
# eopod configure --tpu-name <YOUR_TPU_NAME>

eopod run pip install math_verify -qU
eopod run pip install ray[default]==2.34.0 -qU
eopod auto-config-ray --self-job
