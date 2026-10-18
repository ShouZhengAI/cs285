# short report for hw1

training steps vs. loss and reward
![](https://raw.githubusercontent.com/ShouZhengAI/blog-images/main/images/train_steps.png)
loss
![](https://raw.githubusercontent.com/ShouZhengAI/blog-images/main/images/train_loss.png)
reward
![](<W&B Chart 10_18_2026, 10_22_30 PM.png>)
push-T video

![](https://raw.githubusercontent.com/ShouZhengAI/blog-images/main/images/20261018222440036.gif)

The MSE policy is a fully connected MLP with **3 hidden layers of 256 units each** and **ReLU** activations. It maps the 5-dimensional observation to a flattened action chunk of `chunk_size × 2` values; with the default chunk size of 8, the output layer has 16 units. We train it with mean squared error on normalized expert action chunks.


# Flow Matching Policy Report

## Training curves

![](https://raw.githubusercontent.com/ShouZhengAI/blog-images/main/images/20261018225929638.png)
The policy was trained for 400 epochs (76,000 optimizer steps), with batch size 128, learning rate 3e-4, and hidden dimensions (256, 256, 256). The plotted loss is a 10-log-point moving average of the minibatch training loss. Evaluation ran every 10,000 steps and used 100 episodes. The final training loss was 0.175; the final mean evaluation reward was **0.844** at step 76,000. The highest recorded evaluation reward was **0.844** at step 76,000.

## Qualitative comparison with MSE

In the inspected final evaluation videos, both policies move the agent to contact and push the T. The flow policy's sampled rollout ends with the T more closely aligned with the green target region, while the inspected MSE rollout ends with a visibly larger offset. This matches the final mean rewards in these runs (flow: 0.844; MSE: 0.635). This is a qualitative observation from the inspected videos, not a claim that every episode behaves this way.

![](https://raw.githubusercontent.com/ShouZhengAI/blog-images/main/images/20261018230055160.png)

![](https://raw.githubusercontent.com/ShouZhengAI/blog-images/main/images/20261018230135588.png)
