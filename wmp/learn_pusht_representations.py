"""
To learn, we use the following losses:
- Consistency loss
- Inverse dynamics loss
- Reward prediction loss

The reward is set to 1 at the end of the episode. Another strategy is to maximize P(task success).
However, considering that many trajectories will take longer than this, instead maybe it will be
good to create reward and Q estimates. Then, we can guide toward higher Q values. I could check out
DynaGuide for some information on how to do this.

It would also be good to create a decoder. Maybe, as well, some way to directly output energies between
states. The energy could take the form of a residual over a predicted ODE from the neural network, but
one which is not necessarily linear so that the network can incorporate constraints. Using a residual
approach may not be ideal, though. Really I just want to create the simplest version that still correctly
represents the distribution of future states we can have, as explicit energy values.

What representations are used? In Dynaguide (and DINO-WM), they use frozen DINO representations, and it's fine. So
I'll stick with this. Dynamics model could be made smaller by only considering stepwise transition probabilities,
because we already implicitly assume that each representation has enough information to predict the dynamics correctly
(through the initial state). Then we have:

p(s_t | s_{t-1}, a_{t-1})

and,

p(a | success)

?

We want to switch from sampling:

p(s, a, s, a, s, a, ... s0 | success)

To the maximum a posteriori:

max_{s, a, ..., s0} p(success | s, a, s, a, s, a, ... s0)

Which is the same as

max_{s, a, ..., s0} log p(success | s, a, s, a, s, a, ... s0)

Or

max_{s, a, ..., s0} log p(s, a, s, a, s, a, ... s0 | success) + log p(success) - log p(s, a, s, a, s, a, ... s0)

p(success) is constant, so we can ignore it, and we have:

max_{s, a, ..., s0} log p(s, a, s, a, s, a, ... s0 | success) - log p(s, a, s, a, s, a, ... s0)

This amounts to maximizing the likelihood of the trajectory given success, while minimizing the likelihood of the trajectory overall. This looks wrong. And in fact, it is; the max occurs over success. To simplify, will use tau to refer to the trajectory...


max_tau p(tau) p(success | tau)

If p(tau) is perfect, then it is an indicator function over valid trajectories, which can only be maximized for valid trajectories. Therefore, we have

max_{valid tau} p(success | tau)

Then, we have:

max_tau log p(tau) + log p(success | tau) =

Assuming the Markov property, p(success | tau) = p(success | s_T), where s_T is the final state in the trajectory. Therefore, we have:

max_tau log p(tau) + log p(success | s_T)

Then log p(success | s_T) = Q(s_T) + Z, proportional to the Q value of the final state. Then, we are maximizing the Q value of the final state of the trajectory subject to a "valid trajectory" constraint. We could also estimate Q through TD-lambda and a learned reward function.

This is also the same as classifier and classifier-free guidance, treating the Q value as an energy function over the trajectory. Or rather, treating the discounted rewards as an energy function.

In classifier and classifier-free guidance, you sort of just hand-wave that this is the case.

"""
