# NFL_big_data_bowl
Transformer based architecture to predict yards gained by a rusher in a given play.

# Task
Given a x,y coordinates, speed, acceleration and some other stats for all players on a given football field predict how many yards the rusher, who did just receive the ball, will gain in the situation. (Actually we predict not a single scalar but rather a probability distribution for whether a rusher will gain at least n yards.)

# Solution
The solution here is inspired by the numerous post competition winning solutions in the kaggle forum. This is just a reimplementation for my own understanding.
Every game situation is rearranged into a 2d vector of size (22, n_features). Most important features are the coordinates and the speed of the players. Then convolutions are applied in order to extract features for every individual player. These embedded player vectors are then fed into a transformer without positional encoding in order to capture the whole game situation context. The positional encoding is omitted because we want a positional invariant game representation. (meaning the model should be oblivious to the order of the players) After the transformer I apply average pooling and finally I use a 2 simple linear output layers to regress the final output distribution.
