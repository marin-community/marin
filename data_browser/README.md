# Data Browser

Marin comes with a data browser that makes it easy to
view datasets (in various formats) and experiments produced by the executor.
The data browser will also allow users to follow links to the Ray dashboard and wandb (for training steps), 
as well as the code that was run to produce the experiment/data.

## Development

To start the data browser:

    docker-compose up --build

And then open http://localhost:5000 in your browser.
