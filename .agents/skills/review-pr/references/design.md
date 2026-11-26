# Review the proposed interface

Review the decision the author is asking maintainers to live with. Assume
the implementation passes its tests. What would the people who use or
extend it still have to know?

Start with the behavior a real consumer needs. Before opening the diff,
sketch a call using the vocabulary of the problem and the base tree. Then
write out how a caller uses the proposed interface. Compare the two on a
second supported use. For each concept the PR introduces, try removing it
from the sketch. What independent decision or constraint would be lost?
Trace each remaining choice to the component that has the information to
make it. This can uncover a design problem even if every branch works.

Challenge your alternative. Follow an existing consumer, deployment or data
contract that might require the proposed separation. Explain the difference
using the actual calls and the work one design saves. A small local change
is a sufficient remedy if it puts the decision in the right place.

Return each distinct, supported design finding with a changed anchor, the
other site that establishes the cost, and the change you recommend. Group
symptoms that would disappear together. Return no design finding when the
alternative only moves the same work. Leave the implementation and test
audit to the main reviewer.
