# pendulum-balancer-rl

this is a single pendulum on a rail with a movable block. the mechanics are to push the block to the left or right and the goal is to get the pendulum upright and balance it there.

everything is written from scratch, no gymnasium and no RL libraries. the physics, the reward, the training loop, and the graphs are all hand-rolled. a very confident thing to say right before deriving equations of motion by hand (cramer's rule, RK4 at 240 Hz, and a special case for when the cart gets pinned against the wall).

there are two modes. easy starts almost upright, so it's just balancing. normal starts hanging straight down, so it has to swing up first, which is where things got interesting (read: stalled). and there are three agents so far: a random baseline that mashes left and right, tabular q-learning, which did solve easy mode, and monte carlo control. every decision and why it was made lives in `notes/decisions.md`, if you want the long version.

```bash
uv run pendulum                                        # no args, pick everything from menus
uv run pendulum play easy                              # you, the arrow keys, and physics
uv run pendulum train bellman --mode normal --episodes 200
uv run pendulum bench results/<run folder>             # interactive replay page, opens from disk
```

each training run lands in `results/<agent>_<mode>_<timestamp>/` with a metrics CSV, a recording, a results graph, and a `bench.html` you can scrub through.
