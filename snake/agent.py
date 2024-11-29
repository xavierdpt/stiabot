import math
import torch
import random
import numpy as np
from collections import deque
from model import Linear_QNet, QTrainer
from helper import plot


MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Game:
    def __init__(self, n_cards):
        self.n_cards=n_cards
        self.reset()
    def print_hand(self,label):
        sep=False
        print(label,end="")
        print(": ",end="")
        for card in self.hand:
            if sep:
                print(", ",end="")
            print(card,end="")
            sep=True
        print()
    def reset(self):
        self.cards=list(range(1,self.n_cards+1))
        self.hand=[]
        for n in range(7):
            card = self.pick_card()
            if card != None:
                self.hand.append(card)
        self.table=[]
        self.score=0
    def pick_card(self):
        if(len(self.cards))==0:
            return None
        idx=math.floor(random.random()*len(self.cards))
        card = self.cards[idx]
        self.cards=self.cards[0:idx]+self.cards[idx+1:]
        return card
    def play_step(self,card):
        if self.is_in_hand(card):
            self.move_to_table(card)
            reward=1
            self.score+=1
        else:
            reward=-10
            self.score-=1
        return reward
    def is_in_hand(self,card):
        for c in self.hand:
            if c == card:
                return True
        return False
    def is_done(self):
        return len(self.hand)==0
    def move_to_table(self,card):
        new_hand = []
        for c in self.hand:
            if c==card:
                self.table.append(card)
            else:
                new_hand.append(c)
        self.hand=new_hand

class Agent:

    def __init__(self, n_cards):
        self.n_cards=n_cards
        self.n_games = 0
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()

        # Whether each card is in stack
        n_inputs = n_cards
        # Whether each card is in hand
        n_inputs += n_cards
        # Whether each card is on table
        n_inputs += n_cards
        # Whether the game is done
        n_inputs+=1

        n_middle = n_cards*10

        # Whether a card can be played
        n_outputs = n_cards

        self.model = Linear_QNet(n_inputs, n_middle, n_outputs)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, game):
        cards = {}
        for card in range(1,game.n_cards+1):
            cards[card]=0 # Cards in deck
        for card in game.hand:
            cards[card]=1 # Cards in hand
        for card in game.table:
            cards[card]=2 # Card on table

        state = [game.is_done()]

        # Cards on deck
        for card in range(1,game.n_cards+1):
            state.append(cards[card]==0)

        # Cards in hand
        for card in range(1,game.n_cards+1):
            state.append(cards[card]==1)

        # Cards on table
        for card in range(1,game.n_cards+1):
            state.append(cards[card]==1)

        return state

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        # Pass the data directly to the trainer
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state, score):
        move = [0]*self.n_cards
        use_model=True
        if(score<0):
            # The worst the agent is, the more we help him with random decision
            threshold = 1-math.exp(score)
            rnd = random.random()
            use_model = threshold < rnd
        if use_model:
            #print("Using model")
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move_index = torch.argmax(prediction).item()
            move[move_index]=1
        else:
            #print("Random choice")
            move[math.floor(random.random()*self.n_cards)]=1
        return move

def train():
    # Next step: Start with 7 cards and wait for the model to learn
    # Then slowly increase the number of cards to 104
    # For every step, wait to have ten score of 7 in a row
    # This way the agent should learn more quickly to play cards from its hand

    total_n_cards = 104
    n_cards = 14
    agent = Agent(n_cards)
    game = Game(n_cards)

    plot_scores = []
    plot_mean_scores = []
    total_score = 0

    record = 0

    print_new_game=True
    while agent.n_games < 300:
        if print_new_game:
            #print("New game")
            game.print_hand("Hand")
            print_new_game=False
        # get the current state
        state_old = agent.get_state(game)

        # get the action to perform from the agent
        move = agent.get_action(state_old, game.score)
        card = move.index(1)+1
        #print("Agent is playing",card)

        # reward and score is only a function of move
        reward = game.play_step(card)
        if reward>0:
            #print("It was a good choice")
            #game.print_hand("New hand")
            pass
        elif reward<0:
            #print("It was a bad choice")
            pass

        

        done = game.is_done()

        # the new state is the same as the previous state because the game has no state
        state_new = agent.get_state(game)

        # train on this particular move
        agent.train_short_memory(state_old, move, reward, state_new, done)

        agent.remember(state_old, move, reward, state_new, done)
        if done:
            score = game.score
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            if score > record:
                record = score
                agent.model.save()
            print('Game', agent.n_games, 'Score', score, 'Record:', record)
            print()
            print_new_game=True
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)



if __name__ == '__main__':
    train()