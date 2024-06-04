#!/bin/sh

SESSION_NAME="marin"
CONDA_ENV="marin"

# Check if Session Exists
SESSION_EXISTS=$(tmux list-sessions | grep $SESSION_NAME)

if [ "$SESSION_EXISTS" = "" ]
then
  # Create a named TMUX session and attach (prevents clobbering)
  tmux new-session -d -s $SESSION_NAME

  # Configure TMUX Panes (70% - 30%)
  # =================================================
  #           shell               |  Ray Dashboard  |
  #   (source conda environment)  |                 |
  # =================================================
  tmux rename-window -t $SESSION_NAME:0 "Ray"
  tmux split-window -t $SESSION_NAME -h
  tmux resize-pane -t $SESSION_NAME:0.0 -x "70%"

  # Source Conda Environment in Panes
  tmux send-keys -t $SESSION_NAME:0.0 "conda activate ${CONDA_ENV}" C-m C-l
  tmux send-keys -t $SESSION_NAME:0.1 "conda activate ${CONDA_ENV}" C-m C-l

  # Launch Ray Dashboard (Right Pane)
  tmux send-keys -t $SESSION_NAME:0.1 "ray dashboard infra/marin-cluster.yaml" C-m

  # Set Active (Left Pane)
  tmux select-pane -t $SESSION_NAME:0.0
fi

# Enter Session
tmux attach -t $SESSION_NAME
