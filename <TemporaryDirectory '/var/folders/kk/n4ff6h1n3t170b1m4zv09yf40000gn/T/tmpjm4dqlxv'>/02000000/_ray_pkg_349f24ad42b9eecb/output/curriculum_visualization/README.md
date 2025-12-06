# Curriculum Learning Visualization

This directory contains comprehensive visualizations of the adaptive curriculum learning system.

## Generated Visualizations

### Figure 1: Plateau Detection (`fig1_plateau_detection.png`)
Shows how plateau detection works with different reward patterns:
- **Flat Rewards**: Quickly plateaus (stable rewards trigger detection)
- **Improving Rewards**: No plateau until reaching target
- **Noisy Stable**: Plateaus despite noise (robust to variance)
- **Sigmoid**: Eventually plateaus after improvement curve flattens

Each panel shows the linear regression fit on the most recent window (red dashed line) and highlights plateaued regions in green.

### Figure 2: Sampling Weights Over Time (`fig2_sampling_weights.png`)
Stacked area chart showing how 5 environments compete for sampling attention across 1000 training steps. Demonstrates:
- Initial exploration of easy lessons
- Progressive unlocking of harder lessons as dependencies are met
- Dynamic reweighting based on success rates
- Graduation removing mastered lessons from the pool

### Figure 3: Lesson Lifecycle Timeline (`fig3_lesson_lifecycle.png`)
Gantt-style timeline showing lesson state transitions:
- **Gray**: Locked (dependencies not met)
- **Green**: Active (unlocked and being sampled)
- **Blue**: Graduated (mastered and deprioritized)
- **Gold diamonds**: Unlock events
- **Blue squares**: Graduation events

### Figure 4: Weight Function (`fig4_weight_function.png`)
Two panels showing the sampling weight mechanics:
- **Left**: Base quadratic weight function (-4s² + 4s) peaking at 50% success
- **Right**: Exploration bonus effect on final weights for different success rates

### Figure 5: Exploration Bonus Decay (`fig5_exploration_bonus.png`)
Shows how the exploration bonus (1.0 + exp(-0.03 * samples)) decays from 2x to 1x over ~150 samples, ensuring new lessons get sufficient initial exploration.

### Figure 6: Dependency Graph (`fig6_dependency_graph.png`)
Directed acyclic graph (DAG) showing lesson dependencies. Edge labels indicate reward thresholds that must be met before downstream lessons unlock.

### Figure 7: Health Metrics Dashboard (`fig7_health_metrics.png`)
Four-panel dashboard tracking curriculum health:
- **Sampling Entropy**: Diversity measure (low values indicate collapse)
- **Effective Lessons**: Inverse Simpson index (how many lessons are meaningfully sampled)
- **Mean Success Rate**: Average performance across active lessons
- **Lesson State Counts**: Number of locked/active/graduated lessons over time

### Figure 8: Training vs Eval (`fig8_training_vs_eval.png`)
Per-lesson comparison of training (solid) vs evaluation (dashed) success rates. Shows:
- Training smoothed success (continuous updates from rollouts)
- Eval smoothed success (periodic evaluations, less frequent)
- Divergence patterns indicating overfitting or underfitting

## Summary Tables (`tables.html`)

HTML tables providing detailed information:

1. **Lesson Configuration**: Initial setup (dependencies, thresholds, plateau params)
2. **Final State Snapshot**: End-of-run statistics for all lessons
3. **Unlock Events**: When and why each lesson was unlocked
4. **Graduation Events**: When and why each lesson graduated

## Usage

To regenerate these visualizations:

```bash
uv run python src/marin/rl/scripts/visualize_curriculum.py
```

The script simulates a 1000-step curriculum run with 5 lessons and varying difficulty patterns.

## Customization

Edit `src/marin/rl/scripts/visualize_curriculum.py` to:
- Change lesson configurations
- Adjust simulation parameters (num_steps, reward patterns)
- Modify visualization styles
- Add new figures or metrics
