from collections import deque
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
import numpy as np
import random


class Agent:
    def __init__(self, state_size, action_size):
        self.weight_backup = "cartpole_weights.hd5"
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.99
        self.exploration_rate = 1.0
        self.exploration_min = 0.1
        self.exploration_decay = 0.01

        self.model = None
        self.checkpointer = ModelCheckpoint(filepath=self.weight_backup,
                                            verbose=0, save_best_only=False)
        self.build_model()

    def build_model(self):
        self.model = Sequential()
        self.model.add(Dense(24, input_shape=(self.state_size,),
                             activation='relu'))
        self.model.add(Dense(24, activation='relu'))
        self.model.add(Dense(self.action_size, activation='linear'))
        self.model.compile(loss='mse', optimizer='adam')

    def load_model(self):
        self.model = load_model(self.weight_backup)
        self.exploration_rate = 0.0

    def act(self, state):
        # Explore randomly
        if random.random() <= self.exploration_rate:
            return random.randrange(self.action_size)
        else:
            action = self.model.predict(state)
            return np.argmax(action[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, sample_batch_size):
        # Need to collect memories first
        batch_size = min(len(self.memory), sample_batch_size)

        # Get a sample replay from emory
        sample_batch = random.sample(self.memory, batch_size)

        X = []
        y = []
        for state, action, reward, next_state, done in sample_batch:
            target = reward
            # Take into account current and future rewards
            if not done:
                target = reward + self.gamma * np.amax(
                                  self.model.predict(next_state)[0])

            target_future = self.model.predict(state)
            target_future[0][action] = target

            X.append(state)
            y.append(target_future)

        self.model.fit(np.array(X).reshape((-1, self.state_size)),
                       np.array(y).reshape((-1, self.action_size)),
                       batch_size=10, epochs=1, verbose=0,
                       callbacks=[self.checkpointer])

        # Make sure that model still experiments after long time
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= 1 - self.exploration_decay
