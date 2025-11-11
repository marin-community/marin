# Tips and FAQ

See also the [Equinox FAQ](https://docs.kidger.site/equinox/faq/)

## [`hax.debug.diagnose_common_issues`][haliax.debug.diagnose_common_issues]

[haliax.debug.diagnose_common_issues][] is a function that will raise an exception if it detects problems with your module.
Currently, we diagnose:

* Reuse of arrays or NamedArrays in a field. [Equinox modules must be trees.](https://docs.kidger.site/equinox/faq/#a-module-saved-in-two-places-has-become-two-independent-copies)
* Use of arrays or NamedArrays in a static field. Static data in JAX/Equinox must be hashable, and arrays are not hashable.

## [`hax.debug.visualize_shardings`][haliax.debug.visualize_shardings]

Use [haliax.debug.visualize_shardings][] to quickly inspect how a PyTree is sharded.
It prints the sharding of each array leaf, including the mapping from named axes
to physical axes for :class:`haliax.NamedArray` leaves.
