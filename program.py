import random
import math

if False:
    import os
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    from numpy import loadtxt
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    model = Sequential()
    model.add(Dense(12, input_shape=(8,), activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

cardValues = [
    1, # 1
    1, # 2
    1, # 3
    1, # 4
    2, # 5
    1, # 6
    1, # 7
    1, # 8
    1, # 9
    3, # 10
    5, # 11
    1, # 12
    1, # 13
    1, # 14
    2, # 15
    1, # 16
    1, # 17
    1, # 18
    1, # 19
    3, # 20
    1, # 21
    5, # 22
    1, # 23
    1, # 24
    2, # 25
    1, # 26
    1, # 27
    1, # 28
    1, # 29
    3, # 30
    1, # 31
    1, # 32
    5, # 33
    1, # 34
    2, # 35
    1, # 36
    1, # 37
    1, # 38
    1, # 39
    3, # 40
    1, # 41
    1, # 42
    1, # 43
    5, # 44
    2, # 45
    1, # 46
    1, # 47
    1, # 48
    1, # 49
    3, # 50
    1, # 51
    1, # 52
    1, # 53
    1, # 54
    6, # 55
    1, # 56
    1, # 57
    1, # 58
    1, # 59
    3, # 60
    1, # 61
    1, # 62
    1, # 63
    1, # 64
    2, # 65
    5, # 66
    1, # 67
    1, # 68
    1, # 69
    3, # 70
    1, # 71
    1, # 72
    1, # 73
    1, # 74
    2, # 75
    1, # 76
    5, # 77
    1, # 78
    1, # 79
    3, # 80
    1, # 81
    1, # 82
    1, # 83
    1, # 84
    2, # 85
    1, # 86
    1, # 87
    5, # 88
    1, # 89
    3, # 90
    1, # 91
    1, # 92
    1, # 93
    1, # 94
    2, # 95
    1, # 96
    1, # 97
    1, # 98
    5, # 99
    3, # 100
    1, # 101
    1, # 102
    1, # 103
    1  # 104
]

nCards = 104

class GameState:
    def __init__(self):
        self.cards=list(range(1,nCards+1))
        self.initHands()
        self.initBoard()
    def initHands(self):
        self.hands = [[],[],[],[]]
        for turn in range(7):
            for handNum in range(4):
                self.hands[handNum].append(self.pickCard())
    def initBoard(self):
        self.boardStacks = [[],[],[],[],[]]
        for stackNum in range(len(self.boardStacks)):
            self.boardStacks[stackNum].append(self.pickCard())
    def pickCard(self):
        index=math.floor(random.random()*len(self.cards))
        card = self.cards[index]
        self.cards=self.cards[:index]+self.cards[index+1:]
        return card;
    def hasCards(self):
        return len(self.cards)>0
    def getCards(self,playerNum):
        return self.hands[playerNum]
    def clearCard(self,playerNum,index):
        self.hands[playerNum]=self.hands[playerNum][:index]+self.hands[playerNum][index+1:]
    def getStacks(self):
        return self.boardStacks
    def canPlay(self):
        for hand in self.hands:
            if len(hand)==0:
                return False;
    def printState(self):
        print("Available cards: ",end="")
        for card in self.cards:
            print(" ",end="")
            print(card, end="")
        print()
        print("Hands:")
        for handNum in range(4):
            print("Player ",end="")
            print(handNum,end="")
            print(": ",end="")
            for card in self.hands[handNum]:
                print(" ",end="");
                print(card,end="")
            print()
        print("Board:")
        for stackNum in range(len(self.boardStacks)):
            print("Stack ",end="")
            print(stackNum,end="")
            print(": ",end="")
            for card in self.boardStacks[stackNum]:
                print(" ",end="")
                print(card,end="")
            print()

class Move:
    def __init(self, card, stack):
        self.card=card
        self.stack=stack

class RandomPlayer:
    def __init__(self,playerNum):
        self.playerNum = playerNum
    def play(self,gameState):
        cards = gameState.getCards(self.playerNum)
        index=math.floor(random.random()*len(cards))
        card = cards[index]
        gameState.clearCard(self.playerNum,index)
        stacks = gameState.getStacks()
        stackNum = math.floor(random.random()*len(stacks))
        return Move(card,stackNum)

players = []
for playerNum in range(4):
    players.append(RandomPlayer(playerNum))

gameState = GameState()

while gameState.canPlay():

gameState.printState()
if False:
    print(len(gameState.cards))
    while gameState.hasCards():
        print(gameState.pickCard()) 

# Inputs:
# - For each card, whether it is in the hand
# - For each card, whether it is on the board, and not at the bottom or at the top
# - For each card, whether it is at the bottom of a stack
# - For each card, whether it is at the top of a stack
# - For each card, whether it as been collected in previous turn

# Outputs:
# - For each card, whether we should play that card
#   - If the model selects a card that is not in the hand, it's a fail
#   - If the model selects a card that is in the hand, it an ok move
# - For each card, whether it's a good candidate to use to collect the stack
#  - If the model select a card that is not at the bottom of a stack, it's a fail
#  - If the model selects a card that is at the bottom of a stack, it's an ok move


# Game starts with 5 cards on the on board, 7 cards in the hand and 4 players
# 3 players use random strategy
# 1 player is the bot, learning

# Each player chooses a card to play
# - Random players select their cards
# - AI bot puts creates its inputs, evaluates its model, and uses the result to select the card

# The games continues until the head and the points are scored
