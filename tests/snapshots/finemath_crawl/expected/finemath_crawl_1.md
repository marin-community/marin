Character stoping distance math

i’m trying to work out the math to give me the stopping distance and stopping time of a character. i know that if it wasn’t for ground friction it would be:
StopTime= Speed/braking Deceleration
Stop Distance = (Speed/StopTime) / 2

but i’m finding it hard to wrap my head around how the ground friction works. it defaults to 8 and stooping times and distances don’t seem to have a linear relation ship to movement speed. so i’m guess there is some exponential thing i’m not understanding.
any help would be awesome,
Thanks

Here is a Tutorial to how to make a wheeled vehicle in Unreal.

In reality vehicle experience 2 types of meaningful friction (static and dynamic) and materials have different friction coefficients for those cases. Here is a Wikipedia article for more information on the physics side of things. Keep in mind that all those calculations already simulated in the Unreal physics engine.

In short:

FrictionForce = FrictionCoefficient * NormalForce

but also

FrictionForce = ObjectMass * FrictionDeceleration

so

FrictionDeceleration = FrictionCoefficient * NormalForce / ObjectMass

from this point on you get

StopTime = Speed / FrictionDeceleration
StopDistance = Speed * StopTime / 2

Keep in mind that the Friction Coefficient must be different when the car is sliding. (Above some maximum FrictionForce the coefficient changes) NormalForce is largely dependent on your car weight and the downforce.

If you need a full physical model of a car you will need to account for the downforce, suspension hardness, loss of friction, weight transfer between wheels, centrifugal force in turns, contact surface temperature and area (between the wheels and the road).

Some of these can and are frequently ignored in games. (temperature and weight transfer) Most of the ones that are needed (like loss of friction and down force) have simplified math models that are used in games but they are in no way simple to understand.

Happy coding

thanks for the reply, very interesting (and useful) stuff. still finding it hard to make sense of the character implementation of friction in relation to this however. mass doesn’t effect stooping distance or time in my tests with the character movement component and its hard to pin down what the FrictionCoefficient NormalForce are in this context.

Character movement component does not use physics for it’s movement.

Although it has some rudimentary calculations (for gravity and acceleration) it intentionally has no friction or complex physics involved. There is a way to make a pawn use the physics engine which, however does not go through the movement component.

Check https://docs-origin.unrealengine.com/latest/INT/Videos/PLZlv_N0_O1ga0aV9jVqJgog0VWz1cLL5f/lFpXqggbUP4/index.html

i’m looking to predict the stopping distance and time of the standard Character movement component. so i guess i’m looking for whatever there approximation of friction is (as the component has a number of variable associated with friction) and it doesn’t stop linearly in the way it would if it had a constant deceleration force applied.

In this case the you should go VERY in-depth with the character movement component.

If I were you I would start with:

void UCharacterMovementComponent::CalcVelocity(...){..} //CharacterMovementComponent.cpp line:2986

which calls:

bool UCharacterMovementComponent::ApplyRequestedMove(...){...} //CharacterMovementComponent.cpp line:3073

and in there is a line of code that states:

Velocity = Velocity - (Velocity - RequestedMoveDir * VelSize) * FMath::Min(DeltaTime * Friction, 1.f);

I guess you can work your way backwards from that point but note that the file is over TEN THOUSAND lines so it wont be easy. The component has too many variables which modify the values given to these functions.

I know I am a bit late for answering, but this is how friction is calculated for ground movement in UE4 character movement component

Velocity = Velocity - ((Velocity * Friction) + Braking Deceleration) * Delta Time

Friction is only used in case of braking its not added when accelerating

Friction = Ground Friction * Braking Friction Factor

if you still searching for calculating stop location I posted my code here: Predict Stop Position of Character ahead in time - Character & Animation - Unreal Engine Forums
