#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:





# In[1]:


from IPython.display import Image


# ## Hex 
# 
# Hex es un juego de estrategia para dos jugadores que se juegan en una cuadrícula hexagonal , teóricamente de cualquier tamaño y varias formas posibles, pero tradicionalmente como un rombo de 11 × 11 . Los jugadores alternan colocando marcadores o piedras en espacios desocupados en un intento de unir sus lados opuestos del tablero en una cadena ininterrumpida. Un jugador debe ganar; No hay sorteos. El juego tiene una estrategia profunda, tácticas agudas y una base matemática profunda relacionada con el teorema del punto fijo de Brouwer
# 
# 

# In[7]:


from IPython.display import Image
Image(url='hex1.jpg')


# In[ ]:





# ## Aprendizaje Reforzado (RL)
# El aprendizaje por refuerzo es una área de la inteligencia artificial que esta centrada en descubrir que acciones se debe tomar para maximizar la señal de recompensa, en otras palabras se centra en como mapear situaciones a acciones que se centren en encontrar dicha recompensa. Al agente no se le dice que acciones tomar, si no al contrario el debe experimentar para encontrar que acciones lo llevan a una mayor recompensa.
# 
# El problema fundamental de de RL es que el agente aprenda a tomar decisiones en un ambiente cambiante, es decir tomar acciones para obtener la mayor reconpensa
# 
# #### Ambiente y estado
# 
# En lugar de ejemplos existe un **ambiente o mundo** el cual podemos observar 
# 
# nuestra perceción del ambiente no siempre es completa
# 
# El ambiente se representa por un vector denominado **estado**
# 
# #### Acciones
# 
# El agente no retorna predicciones sino que toma decisiones
# 
# En cada instante el agente escoge y realiza una acción
# 
# Existen consecuencias, las acciones realizadas pueden modificar el ambiente
# 
# #### Rencompensa
# La retroalimentación del agente no proviene de etiquetas sino de una señal numérica escalar llamada recompensa
# 
# La recompensa está asociada a uno o más estados
# 
# La recompensa puede ser positiva o negativa
# 
# #### Diferencias claves
# Supervisión:Al agente no se le dice que acción es buena, sino que estados son buenos
# 
# Prueba y error: El agente debe descubrir que acción le entrega la mayor recompensa probándolas una a una
# 
# Temporalidad: El entrenamiento y la ejecución son secuenciales, no se puede asumir iid
# 
# Retraso en la retroalimentación: Las recompensas pueden demorar en llegar, las acciones pueden no traer recompensa inmediata pero si en el futuro

# In[8]:


Image(url='RL.jpg')


# In[ ]:





# In[ ]:





# ## hex_zero_model
# 
# Se construye en modelo de  la Red Neuronal profunda que más adelante se utiliza para la predicción de las políticas y valores

# In[3]:



from keras.models import Sequential

model = Sequential()


# In[34]:


from keras.engine.topology import Input
from keras.engine.training import Model
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation, Dense, Flatten
from keras.layers.merge import Add
from keras import optimizers
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.models import load_model ### para guardar los modelos 


# In[5]:


cnn_filter_num = 128
cnn_first_filter_size = 2
cnn_filter_size = 2
l2_reg = 0.0001
res_layer_num = 20
n_labels = 64
value_fc_size = 64
learning_rate = 0.1 # schedule dependent on thousands of steps, every 200 thousand steps, decrease by factor of 10
momentum = 0.9


def build_model():
    """
    Builds the full Keras model and returns it.
    """
    in_x = x = Input((1, 8, 8))

    # (batch, channels, height, width)
    x = Conv2D(filters=cnn_filter_num,   kernel_size=cnn_first_filter_size, padding="same", data_format="channels_first", use_bias=False, kernel_regularizer=l2(l2_reg), name="input_conv-"+str(cnn_first_filter_size)+"-"+str(cnn_filter_num))(x)
    x = BatchNormalization(axis=1, name="input_batchnorm")(x)
    x = Activation("relu", name="input_relu")(x)

    for i in range(res_layer_num):
        x = _build_residual_block(x, i + 1)

    res_out = x

    # for policy output
    x = Conv2D(filters=2, kernel_size=1, data_format="channels_first", use_bias=False, kernel_regularizer=l2(l2_reg), name="policy_conv-1-2")(res_out)

    x = BatchNormalization(axis=1, name="policy_batchnorm")(x)
    x = Activation("relu", name="policy_relu")(x)
    x = Flatten(name="policy_flatten")(x)

    # no output for 'pass'
    policy_out = Dense(n_labels, kernel_regularizer=l2(l2_reg), activation="softmax", name="policy_out")(x)

    # for value output
    x = Conv2D(filters=4, kernel_size=1, data_format="channels_first", use_bias=False, kernel_regularizer=l2(l2_reg), name="value_conv-1-4")(res_out)

    x = BatchNormalization(axis=1, name="value_batchnorm")(x)
    x = Activation("relu",name="value_relu")(x)
    x = Flatten(name="value_flatten")(x)
    x = Dense(value_fc_size, kernel_regularizer=l2(l2_reg), activation="relu", name="value_dense")(x)

    value_out = Dense(1, kernel_regularizer=l2(l2_reg), activation="tanh", name="value_out")(x)

    model = Model(in_x, [policy_out, value_out], name="hex_model")

    sgd = optimizers.SGD(lr=learning_rate, momentum=momentum)

    losses = ['categorical_crossentropy', 'mean_squared_error']

    model.compile(loss=losses, optimizer='adam', metrics=['accuracy', 'mae'])

    model.summary()
    return model

def _build_residual_block(x, index):
    in_x = x
    res_name = "res"+str(index)
    x = Conv2D(filters=cnn_filter_num, kernel_size=cnn_filter_size, padding="same", data_format="channels_first", use_bias=False, kernel_regularizer=l2(l2_reg), name=res_name+"_conv1-"+str(cnn_filter_size)+"-"+str(cnn_filter_num))(x)
    x = BatchNormalization(axis=1, name=res_name+"_batchnorm1")(x)
    x = Activation("relu",name=res_name+"_relu1")(x)
    x = Conv2D(filters=cnn_filter_num, kernel_size=cnn_filter_size, padding="same", data_format="channels_first", use_bias=False, kernel_regularizer=l2(l2_reg), name=res_name+"_conv2-"+str(cnn_filter_size)+"-"+str(cnn_filter_num))(x)
    x = BatchNormalization(axis=1, name="res"+str(index)+"_batchnorm2")(x)
    x = Add(name=res_name+"_add")([in_x, x])
    x = Activation("relu", name=res_name+"_relu2")(x)
    return x


# In[ ]:





# ## sl.bootstrap
#  contiene un script para arrancar la red neuronal en **hex_data** existentes, llamando a hex_zero_model para construir la red neuronal antes de entrenar la red neuronal para las épocas especificadas
# A partir de los datos ya existentes **hex_data** arranca la red neuronal y la entrena con 25 epoch 
# 
# # no compilar

# In[9]:


import numpy as np


def load_data(filename):
    hex_data = np.load(filename)

    states = hex_data['states']
    turns = hex_data['turns']
    visits = hex_data['visits']
    moves = hex_data['moves']
    values = hex_data['values']

    for i in range(states.shape[0]):
        if turns[i] == -1:
            states[i] = states[i].T
            moves[i] = np.array([[moves[i][1], moves[i][0]]])
            visits[i] = visits[i].T
            values[i] = 1 - values[i].T

    # reshape data for model (channels first)
    states = states.reshape(states.shape[0], 1, 8, 8)

    # train_X = states[:4*states.shape[0] // 5]
    # test_X = states[4*states.shape[0] // 5:]
    train_X = states

    probabilities = calculate_probabilities(visits)
    y_values = calculate_values(moves, values)

    training_probs = probabilities[:4*probabilities.shape[0] // 5]
    training_values = y_values[:4*y_values.shape[0] // 5]
    testing_probs = probabilities[4*y_values.shape[0] // 5:]
    testing_values = y_values[4*y_values.shape[0] // 5:]
    
    train_Y = {'policy_out':probabilities, 'value_out':y_values}
    # test_Y = {'policy_out':testing_probs, 'value_out':testing_values}

    return train_X, train_Y

def calculate_probabilities(visits):
    normalize_sums = visits.sum(axis=1).sum(axis=1)
    reshaped = visits.reshape((visits.shape[0], visits.shape[1]*visits.shape[2]))

    normalized = reshaped/normalize_sums[:,None]

    probabilities = normalized.reshape((visits.shape[0], visits.shape[1]*visits.shape[2]))

    return probabilities

def calculate_values(moves, values):
    y_values = np.array([value[move[0]][move[1]] for move, value in zip(moves, values)])
    return y_values


train_X, train_Y = load_data('hex_data.npz')
model = build_model()
history = model.fit(train_X, train_Y, verbose = 1, validation_split=0.2, epochs = 25, shuffle=True)

# loss, accuracy = model.evaluate(test_X, test_Y, verbose = 1)
# print("accuracy: {}%".format(accuracy*100))

model.save('new_supervised_zero.h5')


# In[35]:


model.save('new_supervised_zero.h5')


# ## BasicPlayers.py

# In[27]:


from random import choice
from sys import stdin

class HumanPlayer:
    """Player that gets moves from command line input."""
    def __init__(self, *args):
        self.name = "Human"

    def getMove(self, game):
        move = None
        while move not in game.availableMoves:
            print("select a row and column")
            try:
                line = input()
                move = (int(line[0]), int(line[1]))
            except ValueError:
                print("invalid move")
            if move not in game.availableMoves:
                print("invalid move")
        return move

class RandomPlayer:
    """Player that selects a random legal move."""
    def __init__(self, *args):
        self.name = "Random"

    def getMove(self, game):
        n= len(game.availableMoves)
        ranmove=choice(range(0,n))
        return game.availableMoves[ranmove]
    


# ## AlphaHex

# In[17]:


from math import log, sqrt
from numpy.random import choice
from numpy import array
import numpy as np
import sys

class Node(object):
    """Node used in MCTS"""
    def __init__(self, state, parent_node, prior_prob):
        self.state = state
        self.children = {} # maps moves to Nodes
        self.visits = 0
        self.value = 0.5
        # self.value = 0.5 if parent_node is None else parent_node.value
        self.prior_prob = prior_prob
        self.prior_policy = np.zeros((8, 8))
        self.parent_node = parent_node

    def updateValue(self, outcome, debug=False):
        """Updates the value estimate for the node's state."""
        if debug:
            print('visits: ', self.visits)
            print('before value: ', self.value)
            print('outcome: ', outcome)
        self.value = (self.visits*self.value + outcome)/(self.visits+1)
        self.visits += 1
        if debug:
            print('updated value:', self.value)
    def UCBWeight_noPolicy(self, parent_visits, UCB_const, player):
        if player == -1:
            return (1-self.value) + UCB_const*sqrt(parent_visits)/(1+self.visits)
        else:
            return self.value + UCB_const*sqrt(parent_visits)/(1+self.visits)
    def UCBWeight(self, parent_visits, UCB_const, player):
        """Weight from the UCB formula used by parent to select a child."""
        if player == -1:
            return (1-self.value) + UCB_const*self.prior_prob/(1+self.visits)
        else:
            return self.value + UCB_const*self.prior_prob/(1+self.visits)

class MCTS:
    def __init__(self, model, UCB_const=2, use_policy=True, use_value=True):
        self.visited_nodes = {} # maps state to node
        self.model = model
        self.UCB_const = UCB_const
        self.use_policy = use_policy
        self.use_value = use_value

    def runSearch(self, root_node, num_searches):
        # start search from root
        for i in range(num_searches):
            selected_node = root_node
            available_moves = selected_node.state.availableMoves
            # if we've already explored this node, continue down path until we reach a node we haven't expanded yet by selecting node w/ largest UCB weight
                # select node that maximizes Upper Confidence Bound
            while len(available_moves) == len(selected_node.children) and not selected_node.state.isTerminal:
                if selected_node == root_node:
                    selected_node = self._select(selected_node, debug=False)
                else:
                    selected_node = self._select(selected_node, debug=False)
                available_moves = selected_node.state.availableMoves
            if not selected_node.state.isTerminal:
                if self.use_policy:
                    if selected_node.state not in self.visited_nodes:
                        selected_node = self.expand(selected_node, debug=False)
                    outcome = selected_node.value
                    if root_node.state.turn == -1:
                        outcome = 1-outcome
                    self._backprop(selected_node, root_node, outcome, debug=False)
                else:
                    moves = selected_node.state.availableMoves
                    np.random.shuffle(moves)
                    for move in moves:
                        if not selected_node.state.makeMove(move) in self.nodes:
                            break
            else:
                outcome = 1 if selected_node.state.winner == 1 else 0
                self._backprop(selected_node, root_node, outcome)
    def create_children(self, parent_node):
        if len(parent_node.state.availableMoves) != len(parent_node.children):
            for move in parent_node.state.availableMoves:
                next_state = parent_node.state.makeMove(move)
                child_node = Node(next_state, parent_node, parent_node.prior_policy[move[0]][move[1]])
                # print(parent_node.prior_policy[move[0]][move[1]])
                parent_node.children[move] = child_node
    def _select(self, parent_node, debug=False):
        '''returns node with max UCB Weight'''
        # print(parent_node.prior_policy)
        # if len(parent_node.state.availableMoves) != len(parent_node.children):
        #    for move in parent_node.state.availableMoves:
        #        next_state = parent_node.state.makeMove(move)
        #        child_node = Node(next_state, parent_node, parent_node.prior_policy[move[0]][move[1]])
        #        # print(parent_node.prior_policy[move[0]][move[1]])
        #        parent_node.children[move] = child_node
        children = parent_node.children
        items = children.items()
        if self.use_policy:
            UCB_weights = [(v.UCBWeight(parent_node.visits, self.UCB_const, parent_node.state.turn), v) for k,v in items]
        else:
            UCB_weights = [(v.UCBWeight_noPolicy(parent_node.visits, self.UCB_const, parent_node.state.turn), v) for k,v in items]
        # if debug:
        #   print([k for k, v in UCB_weights])
        # sys.exit(1)
        # choose the action with max UCB
        node = max(UCB_weights, key=lambda c: c[0])
        if debug:
            print('weight:', node[0])
            print('move:', node[1].state)
            print('value:', node[1].value)
            print('visits:', node[1].visits)
        return node[1]

    def modelPredict(self, state):
        if state.turn == -1:
            board = (-state.board).T.reshape((1, 1, 8, 8))
        else:
            board = state.board.reshape((1, 1, 8, 8))
        probs, value = self.model.predict(board)
        value = value[0][0]
        probs = probs.reshape((8, 8))
        if state.turn == -1:
            probs = probs.T
        return probs, value

    def expandRoot(self, state):
        root_node = Node(state, None, 1)
        if self.use_policy or self.use_value:
            probs, value = self.modelPredict(state)
            root_node.prior_policy = probs
        if not self.use_value:
            value = self._simulate(root_node)
        root_node.value = value
        self.visited_nodes[state] = root_node
        self.create_children(root_node)
        return root_node

    def expand(self, selected_node, debug=False):
        # policy = [selected_node.prior_policy[move] for move in selected_node.state.availableMoves]
        # move = selected_node.state.availableMoves[policy.index(max(policy))]
        # next_state = selected_node.state.makeMove(move)
        # child_node = Node(next_state, selected_node, selected_node.prior_policy[move])
        if self.use_policy or self.use_value:
            probs, value = self.modelPredict(selected_node.state)
            selected_node.prior_policy = probs
        if not self.use_value:
            # select randomly
            value = self._simulate(selected_node)
        if debug:
            print('expanding node', selected_node.state)
        selected_node.value = value
        self.visited_nodes[selected_node.state] = selected_node
        self.create_children(selected_node)
        return selected_node





    def _simulate(self, next_node):
        # returns outcome of simulated playout
        state = next_node.state
        while not state.isTerminal:
            available_moves = state.availableMoves
            index = choice(range(len(available_moves)))
            move = available_moves[index]
            state = state.makeMove(move)
        return (state.winner + 1) / 2

    def _backprop(self, selected_node, root_node, outcome, debug=False):
        current_node = selected_node
        # print(outcome)
        if selected_node.state.isTerminal:
            outcome = 1 if selected_node.state.winner == 1 else 0
        while current_node != root_node:
            if debug:
                print('selected_node: ', selected_node.state)
                print('outcome: ', outcome)
                print('backpropping')
            current_node.updateValue(outcome, debug=False)
            current_node = current_node.parent_node
            # print(current_node.visits)
        # update root node
        root_node.updateValue(outcome)

    def getSearchProbabilities(self, root_node):
        children = root_node.children
        items = children.items()
        child_visits = [child.visits for action, child in items]
        sum_visits = sum(child_visits)
        # print(child_visits)
        if sum_visits != 0:
            normalized_probs = {action: (child.visits/sum_visits) for action, child in items}
        else:
            normalized_probs = {action: (child.visits/len(child_visits)) for action, child in items}
        return normalized_probs

class DeepLearningPlayer:
    def __init__(self, model, rollouts=1600, save_tree=True, competitive=False):
        self.name = "AlphaHex"
        self.bestModel = model
        # self.player = player
        self.rollouts = rollouts
        self.MCTS = None
        self.save_tree = save_tree
        self.competitive = competitive
    def getMove(self, game):
        if self.MCTS is None or not self.save_tree:
            self.MCTS = MCTS(self.bestModel)
        if self.save_tree and game in self.MCTS.visited_nodes:
            root_node = self.MCTS.visited_nodes[game]
        else:
            root_node = self.MCTS.expandRoot(game)
        self.MCTS.runSearch(root_node, self.rollouts)
        searchProbabilities = self.MCTS.getSearchProbabilities(root_node)
        moves = list(searchProbabilities.keys())
        probs = list(searchProbabilities.values())
        prob_items = searchProbabilities.items()
        print(probs)
        # if competitive play, choose highest prob move
        if self.competitive:
            best_move = max(prob_items, key=lambda c: c[1])
            print(best_move)
            # sys.exit(1)
            return best_move[0]
        # else if self-play, choose stochastically
        else:
            chosen_idx = choice(len(moves), p=probs)
            return moves[chosen_idx]


# In[41]:


from MonteCarloTreeSearch import MCTSPlayer


# ## Hex.py

# In[23]:


import numpy as np
from scipy.ndimage import label
from keras.models import load_model
import sys

_adj = np.ones([3,3], int)
_adj[0,0] = 0
_adj[2,2] = 0

RED   = u"\033[1;31m"
BLUE  = u"\033[1;34m"
RESET = u"\033[0;0m"
CIRCLE = u"\u25CF"

RED_DISK = RED + CIRCLE + RESET
BLUE_DISK = BLUE + CIRCLE + RESET
EMPTY_CELL = u"\u00B7"

RED_BORDER = RED + "-" + RESET
BLUE_BORDER = BLUE + "\\" + RESET

def print_char(i):
    if i > 0:
        return BLUE_DISK
    if i < 0:
        return RED_DISK
    return EMPTY_CELL

class HexGame:

    def __init__(self, size=8):
        self.size = size
        self.turn = 1
        self.board = np.zeros([size, size], int)

        self._moves = None
        self._terminal = None
        self._winner = None
        self._repr = None
        self._hash = None

    def __repr__(self):
        if self._repr is None:
            self._repr = u"\n" + (" " + RED_BORDER)*self.size +"\n"
            for i in range(self.size):
                self._repr += " " * i + BLUE_BORDER + " "
                for j in range(self.size):
                    self._repr += print_char(self.board[i,j]) + " "
                self._repr += BLUE_BORDER + "\n"
            self._repr += " "*(self.size) + " " + (" " + RED_BORDER) * self.size
        return self._repr

    def __hash__(self):
        if self._hash is None:
            self._hash = hash(repr(self))
        return self._hash

    def __eq__(self, other):
        return repr(self) == repr(other)

    def makeMove(self, move):
        """Returns a new ConnectionGame in which move has been played.
        A move is a column into which a piece is dropped."""
        hg = HexGame(self.size)
        hg.board = np.array(self.board)
        hg.board[move[0], move[1]] = self.turn
        hg.turn = -self.turn
        return hg

    @property
    def availableMoves(self):
        if self._moves is None:
            self._moves = list(zip(*np.nonzero(np.logical_not(self.board))))
        return self._moves

    @property
    def isTerminal(self):
        if self._terminal is not None:
            return self._terminal
        if self.turn == 1:
            clumps = label(self.board < 0, _adj)[0]
        else:
            clumps = label(self.board.T > 0, _adj)[0]
        spanning_clumps = np.intersect1d(clumps[0], clumps[-1])
        self._terminal = np.count_nonzero(spanning_clumps)
        return self._terminal

    @property
    def winner(self):
        if self.isTerminal:
            return -self.turn
        return 0



from MonteCarloTreeSearch import MCTSPlayer
# from AlphaHex import  DeepLearningPlayer

players = {"random":RandomPlayer,
           "human":HumanPlayer,
#           "mcts":MCTSPlayer,
           # "bryce":HexPlayerBryce,
           "drl":DeepLearningPlayer}

from argparse import ArgumentParser

def play_game(game, player1, player2, show=False):
    """Plays a game then returns the final state."""
    while not game.isTerminal:
        if show:
            print(game)
        if game.turn == 1:
            m = player1.getMove(game)
        else:
            m = player2.getMove(game)
        if m not in game.availableMoves:
            raise Exception("invalid move: " + str(m))
        game = game.makeMove(m)
    if show:
        print(game, "\n")
    print("player", print_char(game.winner), "(", end='')
    print((player1.name if game.winner == 1 else player2.name)+") wins")
    return game

def playBryce(current_model, num_games=10, num_rollouts_1=400, num_rollouts_2=400, play_first=True, show=True):
    for i in range(num_games):
        print('Game #: ' + str(i))
        g = HexGame(8)
        if i%2:
            player1 = DeepLearningPlayer(current_model, rollouts=num_rollouts_1, save_tree=True, competitive=True)
            player2 = HexPlayerBryce(rollouts=num_rollouts_2)
        else:
            player2 = DeepLearningPlayer(current_model, rollouts=num_rollouts_1, save_tree=True, competitive=True)
            player1 = HexPlayerBryce(rollouts=num_rollouts_2)
        # player2 = DeepLearningPlayer(current_model)
        game = play_game(g, player1, player2, show)

def playSelf(current_model, num_games=10, num_rollouts_1=400, num_rollouts_2=400, play_first=True, show=True):
      for i in range(num_games):
          print('Game #: ' + str(i))
          g = HexGame(8)
          player1 = DeepLearningPlayer(current_model, rollouts=num_rollouts_1, save_tree=True)
          player2 = DeepLearningPlayer(current_model, rollouts=num_rollouts_2, save_tree=True)
          game = play_game(g, player1, player2, show)

def playRandom(current_model, num_games=10, num_rollouts=400, play_first=True, show=True):
    for i in range(num_games):
        print('Game #: ' + str(i))
        g = HexGame(8)
        if play_first:
            player1 = DeepLearningPlayer(current_model, rollouts=num_rollouts_1, save_tree=True, competitive=True)
            player2 = RandomPlayer()
        else:
            player1 = RandomPlayer()
            player2 = DeepLearningPlayer(current_model, rollouts=num_rollouts_1, save_tree=True, competitive=True)
        game = play_game(g, player1, player2, show)

def selfPlay(model_a, model_b, num_games, num_rollouts_1, num_rollouts_2, show):
    wins_a = 0
    for i in range(num_games):
        print('Game #: ' + str(i))
        g = HexGame(8)
        player1 = DeepLearningPlayer(model_a, rollouts=num_rollouts_1, save_tree=True, competitive=True)
        player2 = DeepLearningPlayer(model_b, rollouts=num_rollouts_2, save_tree=True, competitive=True)
        if i%2:
            game = play_game(g, player1, player2, show)
            if(game.winner == 1):
                print('a wins')
                wins_a += 1
            else:
                print('b wins')
        else:
            game = play_game(g, player2, player1, show)
            if(game.winner == -1):
                print('a wins')
                wins_a += 1
            else:
                print('b wins')
    print('model a wins: ' + str(wins_a))

if __name__ == "__main__":
    p = ArgumentParser()
    #current_model = load_model('new_supervised_zero.h5')
    # playBryce(current_model, 10, 200, 200, True, False)
    #playSelf(current_model, 10, 400, 400, False, show=True)
    #playBryce(current_model, 20, 300, 300, True, False)
    # playBryce(current_model, 10, 200, 200, False, True)
    # playRandom(current_model, 10, 400, True, True)


# In[26]:


play_game(HexGame(6),RandomPlayer(),RandomPlayer(),show=True)


# In[28]:


play_game(HexGame(3),HumanPlayer(),RandomPlayer(),show=True)


# ## TestAlphaHex.py

# In[29]:



from math import log, sqrt
from numpy.random import choice
from numpy import array
import numpy as np

class Node(object):
  """Node used in MCTS"""
  def __init__(self, state, parent_node, prior_prob):
      self.state = state
      self.children = {} # maps moves to Nodes
      self.visits = 0
      self.value = 0
      self.prior_prob = prior_prob
      self.prior_policy = np.zeros((8, 8))
      self.parent_node = parent_node

  def updateValue(self, outcome):
      """Updates the value estimate for the node's state."""
      self.value = (self.visits*self.value + outcome)/(self.visits+1)
      self.visits += 1
  def UCBWeight_noPolicy(self, parent_visits, UCB_const, player):
      if player == -1:
          return (1-self.value) + UCB_const*sqrt(parent_visits)/(1+self.visits)
      else:
          return self.value + UCB_const*sqrt(parent_visits)/(1+self.visits)
  def UCBWeight(self, parent_visits, UCB_const, player):
      """Weight from the UCB formula used by parent to select a child."""
      if player == -1:
          return (1-self.value) + UCB_const*self.prior_prob*sqrt(parent_visits)/(1+self.visits)
      else:
          return self.value + UCB_const*self.prior_prob*sqrt(parent_visits)/(1+self.visits)

class MCTS:
  def __init__(self, model, UCB_const=1, use_policy=True, use_value=True):
      self.visited_nodes = {} # maps state to node
      self.model = model
      self.UCB_const = UCB_const
      self.use_policy = use_policy
      self.use_value = use_value

  def runSearch(self, root_node, num_searches):
      # start search from root
      for i in range(num_searches):
          selected_node = root_node
          print(selected_node.children)
          available_moves = selected_node.state.availableMoves
          # if we've already explored this node, continue down path until we reach a node we haven't expanded yet by selecting node w/ largest UCB weight
          while len(available_moves) == len(selected_node.children) and not selected_node.state.isTerminal:
              # select node that maximizes Upper Confidence Bound
              selected_node = self._select(selected_node)
              available_moves = selected_node.state.availableMoves
          if not selected_node.state.isTerminal:
              if self.use_policy:
              # expansion
                  actual_policy = []
                  for move in selected_node.state.availableMoves:
                      actual_policy.append(selected_node.prior_policy[move])
                      next_state = selected_node.state.makeMove(move)
                      child_node = self.expand(next_state, selected_node.prior_policy[move], selected_node)
                      selected_node.children[move] = child_node
                      outcome = child_node.value
                      selected_node = child_node
                      self._backprop(selected_node, root_node, outcome)
                  probs, value = self.modelPredict(next_state)
                  move = selected_node.state.availableMoves[actual_policy.index(max(actual_policy))]
              else:
                  moves = selected_node.state.availableMoves
                  np.random.shuffle(moves)
                  for move in moves:
                      if not selected_node.state.makeMove(move) in self.nodes:
                          break
          else:
              outcome = 1 if selected_node.state.winner == 1 else 0
              self._backprop(selected_node, root_node, outcome)

  def modelPredict(self, state):
      if state.turn == -1:
          board = (-1*state.board).T.reshape((1, 1, 8, 8))
      else:
          board = state.board.reshape((1, 1, 8, 8))
      if self.use_policy or self.use_value:
          probs, value = self.model.predict(board)
          value = value[0][0]
          probs = probs.reshape((8, 8))
          if state.turn == -1:
              probs = probs.T
      return probs, value
  def expand(self, state, prior_prob, parent):
      child_node = Node(state, parent, prior_prob)
      if child_node.state.turn == -1:
          board = (-1*child_node.state.board).T.reshape((1, 1, 8, 8))
      else:
          board = child_node.state.board.reshape((1, 1, 8, 8))
      if self.use_policy or self.use_value:
          probs, value = self.model.predict(board)
          value = value[0][0]
          probs = probs.reshape((8, 8))
          if child_node.state.turn == -1:
              probs = probs.T
          child_node.prior_policy = probs
      if not self.use_value:
          value = self._simulate(child_node)
      child_node.value = value
      self.visited_nodes[state] = child_node
      return child_node

  def _select(self, parent_node):
      '''returns node with max UCB Weight'''
      children = parent_node.children
      items = children.items()
      if not self.use_policy:
          UCB_weights = [(v.UCBWeight(parent_node.visits, self.UCB_const, parent_node.state.turn), v) for k,v in items]
      else:
          UCB_weights = [(v.UCBWeight_noPolicy(parent_node.visits, self.UCB_const, parent_node.state.turn), v) for k,v in items]
      # choose the action with max UCB
      node = max(UCB_weights, key=lambda c: c[0])
      return node[1]



  def _simulate(self, next_node):
      # returns outcome of simulated playout
      state = next_node.state
      while not state.isTerminal:
          available_moves = state.availableMoves
          index = choice(range(len(available_moves)))
          move = available_moves[index]
          state = state.makeMove(move)
      return (state.winner + 1) / 2

  def _backprop(self, selected_node, root_node, outcome):
      current_node = selected_node
      # print(outcome)
      if selected_node.state.isTerminal:
          outcome = 1 if selected_node.state.winner == 1 else 0
      while current_node != root_node:
          current_node.updateValue(outcome)
          current_node = current_node.parent_node
          # print(current_node.visits)
      # update root node
      root_node.updateValue(outcome)

  def getSearchProbabilities(self, root_node):
      children = root_node.children
      items = children.items()
      child_visits = [child.visits for action, child in items]
      sum_visits = sum(child_visits)
      if sum_visits != 0:
          normalized_probs = {action: (child.visits/sum_visits) for action, child in items}
      else:
          normalized_probs = {action: (child.visits/len(child_visits)) for action, child in items}
      return normalized_probs

class DeepLearningPlayer:
  def __init__(self, model, rollouts=1600, save_tree=True, competitive=False):
      self.name = "AlphaHex"
      self.bestModel = model
      self.rollouts = rollouts
      self.MCTS = None
      self.save_tree = save_tree
      self.competitive = competitive
  def getMove(self, game):
      if self.MCTS is None:
          self.MCTS = MCTS(self.bestModel)
      if game in self.MCTS.visited_nodes:
          root_node = self.MCTS.visited_nodes[game]
      else:
          root_node = self.MCTS.expand(game, 1, None)
      self.MCTS.runSearch(root_node, self.rollouts)
      searchProbabilities = self.MCTS.getSearchProbabilities(root_node)
      moves = list(searchProbabilities.keys())
      probs = list(searchProbabilities.values())
      prob_items = searchProbabilities.items()
      print(probs)
      # if competitive play, choose highest prob move
      if self.competitive:
          best_move = max(prob_items, key=lambda c: c[1])
          return best_move[0]
      # else if self-play, choose stochastically
      else:
          chosen_idx = choice(len(moves), p=probs)
          return moves[chosen_idx]


# In[38]:


current_model = load_model('new_supervised_zero.h5')


# ## TrainAlphaHexZero.py

# In[40]:



from keras.models import load_model
import numpy as np

def formatTrainingData(training_data):
    """ training data is an array of tuples (boards, probs, value), we need to reshape into np array of state boards for x, and list of two np arrays of search probs and value for y"""
    x = []
    y_values = []
    y_probs = []
    for (board, probs, value) in training_data:
        x.append(board)
        y_probs.append(probs)
        y_values.append(value)

    # use subset of training data
    train_x = np.array(x).reshape((len(x), 1, 8, 8))
    train_y = {'policy_out': np.array(y_probs).reshape((len(y_probs), 64)), 'value_out': np.array(y_values)}
    return train_x, train_y

def reshapedSearchProbs(search_probs):
    moves = list(search_probs.keys())
    probs = list(search_probs.values())
    reshaped_probs = np.zeros(64).reshape(8,8)
    for move, prob in zip(moves, probs):
        reshaped_probs[move[0]][move[1]] = prob
    return reshaped_probs.reshape(64)

def trainModel(current_model, training_data, iteration):
    new_model = current_model
    train_x, train_y = formatTrainingData(training_data)
    np.savez('training_data_'+str(iteration), train_x, train_y['policy_out'], train_y['value_out'])
    #TODO: save training data to npz
    new_model.fit(train_x, train_y, verbose = 1, validation_split=0.2, epochs = 10, shuffle=True)
    new_model.save('new_model_iteration_' + str(iteration) + '.h5')
    return new_model

def evaluateModel(new_model, current_model, iteration):
    numEvaluationGames = 40
    newChallengerWins = 0
    threshold = 0.55

    # play 400 games between best and latest models
    for i in range(int(numEvaluationGames//2)):
        g = HexGame(8)  
        game, _ = play_game(g, DeepLearningPlayer(new_model, rollouts=400), DeepLearningPlayer(current_model, rollouts=400), False)
        if game.winner:
            newChallengerWins += game.winner
    for i in range(int(numEvaluationGames//2)):
        g = HexGame(8)
        game, _ = play_game(g, DeepLearningPlayer(current_model, rollouts=400), DeepLearningPlayer(new_model, rollouts=400), False)
        if game.winner == -1:
            newChallengerWins += game.winner
    winRate = newChallengerWins/numEvaluationGames
    print('evaluation winrate' + str(winRate))
    text_file = open("evaluation_results.txt", "w")
    text_file.write("Evaluation results for iteration" + str(iteration) + ": " + str(winRate) + '\n')
    text_file.close()
    if winRate >= threshold:
        new_model.save('current_best_model.h5')

def play_game(game, player1, player2, show=True):
    """Plays a game then returns the final state."""
    new_game_data = []
    while not game.isTerminal:
        if show:
            print(game)
        if game.turn == 1:
            m = player1.getMove(game)
        else:
            m = player2.getMove(game)
        if m not in game.availableMoves:
            raise Exception("invalid move: " + str(m))
        node = player1.MCTS.visited_nodes[game]
        if game.turn == 1:
            search_probs = player1.MCTS.getSearchProbabilities(node)
            board = game.board
        if game.turn == -1:
            search_probs = player2.MCTS.getSearchProbabilities(node)
            board = -game.board.T
        reshaped_search_probs = reshapedSearchProbs(search_probs)    
        if game.turn == -1:
            reshaped_search_probs = reshaped_search_probs.reshape((8,8)).T.reshape(64)

        if np.random.random() > 0.5:
            new_game_data.append((board, reshaped_search_probs, None))
        if np.random.random() > 0.5:
            new_game_data.append((board, reshaped_search_probs, None))
        game = game.makeMove(m)
    if show:
        print(game, "\n")

        if game.winner != 0:
            print("player", print_char(game.winner), "(", end='')
            print((player1.name if game.winner == 1 else player2.name)+") wins")
        else:
            print("it's a draw")
    outcome = 1 if game.winner == 1 else 0
    new_training_data = [(board, searchProbs, outcome) for (board, searchProbs, throwaway) in new_game_data]
    # add training data
    # training_data += new_training_data
    return game, new_training_data

def selfPlay(current_model, numGames, training_data):
    for i in range(numGames):
        print('Game #: ' + str(i))
        g = HexGame(8)
        player1 = DeepLearningPlayer(current_model, rollouts=400)
        player2 = DeepLearningPlayer(current_model, rollouts=400)
        # player2 = DeepLearningPlayer(current_model)
        game, new_training_data = play_game(g, player1, player2, False)
        training_data+= new_training_data
    return training_data

for i in range(10):
    training_data = []
    current_model = load_model('new_supervised_zero.h5')
    training_data = selfPlay(current_model, 100, training_data)
    new_model = trainModel(current_model, training_data, i)
    evaluateModel(new_model, current_model, i)


# In[ ]:





# Referencias
# https://github.com/magister-informatica-uach/INFO267/blob/master/unidad4/1_fundamentos.ipynb
# https://towardsdatascience.com/hex-creating-intelligent-opponents-with-minimax-driven-ai-part-1-%CE%B1-%CE%B2-pruning-cc1df850e5bd#9995
