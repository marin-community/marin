# Controller V1

# Auth and User workflow

First version, like Ray:

* ssh into bridge machine
* proxy to controller port
* use localhost:<controller port>

No real auth for v0.

# Auth workflow

Controller opens a public port, accessible without SSH.
Users have either a GCP secret or auth via e.g. Google SSO to the controller, get a session token.
Now all controller RPCs accept session token to auth user.
Controller is always at a fixed address or DNS or something like that.

`cluster.marin.community`

GcpResolver -> tag="fluster.controller" -> host:port

# User workflow

run "train" on v5p:4x4 somewhere

# Worker and controller serialization

We should run e..g the dashboard off of the serialized state of the controller/worker
This would be good to use for post-mortem

# Running under appengine would simplify SSO
