import tensorflow as tf
from tensorflow.keras.layers import Dense,Flatten,Conv2D,MaxPool2D
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT,COMPLEX_MOVEMENT
import gym_super_mario_bros
from collections import deque
import random
import numpy as np

MEMORY_SIZE=10_000
BATCH_SIZE=100
MODEL=[]
iteration_count=0
ITERATION_REPLACE=19
epsilon=1
MIN_EPSILON=0.001
EPSILON_DECAY=0.99
target_count=0
TARGET_REPLACE=20
LR=0.001
class Agent():
	 def __init__(self,gamma,input_shape):
	 	self.input_shape=input_shape
	 	self.env=self.create_env()
	 	self.action_space = self.env.action_space.n
	 	self.gamma=gamma
	 	self.model=self.create_model()
	 	self.target=self.create_model()
	 	self.target.set_weights(self.model.get_weights())
	 	self.memory=deque(maxlen=MEMORY_SIZE)
	 	self.flag_got=False
	 def create_env(self):
	 	env = gym_super_mario_bros.make("SuperMarioBros-v0")
	 	env = JoypadSpace(env,COMPLEX_MOVEMENT)
	 	return env
	 
	 def create_model(self):
	 	model = tf.keras.Sequential()
	 	model.add(Conv2D(64,(4,4),input_shape=self.input_shape,activation='relu'))
	 	model.add(MaxPool2D())
	 	model.add(Conv2D(64,(4,4),activation='relu'))
	 	model.add(Flatten())
	 	model.add(Dense(64,activation='relu'))
	 	model.add(Dense(64,activation='relu'))
	 	model.add(Dense(self.action_space,activation='linear'))
	 	model.compile(loss='mse',optimizer=tf.keras.optimizers.Adam(lr=LR))
	 	return model
	 def env_step(step,action):
	 	state_=np.zeros((256,240,4))
	 	for i in range(4):
	 		state,reward,info,done=env.step(action)
	 		state=np.dot(state,[0.2989, 0.5870, 0.1140])/255
	 		state_[:,:,i]=state[:,:]
	 	return state_,reward,info,done
	 def env_reset():
	 	state_=np.zeros((256,240,4))
	 	state=self.env.reset()
	 	state=np.dot(state,[0.2989, 0.5870, 0.1140])/255
	 	for i in range(4):
	 		state_[:,:,i]=state[:,:]
	 	return state_
	 def train(self):
	 	global target_count
	 	if(len(self.memory)<BATCH_SIZE):
	 		return
	 	replay=random.sample(self.memory,BATCH_SIZE)
	 	q_list=self.target.predict(np.array([i[0] for i in replay]))
	 	q_next_list=self.target.predict(np.array([i[3] for i in replay]))
	 	X=[]
	 	for i,(state,action,reward,next_state,done) in enumerate(replay):
	 		if(not done):
	 			q_list[i][action]=reward+self.gamma*np.max(q_next_list[i])
	 		else:
	 			q_list[i][action]=reward
	 		X.append(state)
	 	print("length",q_list.shape)
	 	self.model.train_on_batch(np.array(X),q_list)
	 	target_count+=1
	 	if(target_count>TARGET_REPLACE):
	 		self.target.set_weights(self.model.get_weights())
	 		target_count=0
	 def save_memory(self):
	 	global epsilon
	 	REWARD=0
	 	state=self.env.reset()
	 	state=np.dot(state,[0.2989, 0.5870, 0.1140])/255
	 	state=state.reshape(256,240,1)
	 	done=False
	 	flag=False
	 	while(not done):
	 		if(np.random.random()>epsilon):
	 			action=np.argmax(self.model.predict(state.reshape(1,*self.input_shape))[0])
	 		else:
	 			action=np.random.randint(0,self.action_space)
	 		prev_state=state
	 		state,reward,done,info=self.env.step(action)
	 		self.env.render()
	 		state=np.dot(state,[0.2989, 0.5870, 0.1140])/255
	 		state=state.reshape(256,240,1)
	 		if(info['flag_get']):
	 			reward=info['time']
	 			flag=True
	 		if(info['time']==100):
	 			done=True
	 		if(info['life']<2):
	 			done=True
	 		self.memory.append([prev_state,action,reward,state,done])
	 		REWARD+=reward
	 	if(epsilon>MIN_EPSILON):
	 		epsilon=max(MIN_EPSILON,epsilon*EPSILON_DECAY)
	 	print(REWARD,"eps",epsilon)
	 	return not flag

	 def play(self,till_flag=True,epoch=1):
	 	global iteration_count
	 	iteration_count+=1
	 	if(till_flag):
	 		while(till_flag):
	 				self.train()
	 				till_flag=self.save_memory()
	 	else:
	 		for _ in range(epoch):
	 			self.train()
	 			till_flag=self.save_memory()
	 	if(not till_flag):
	 		print("flag_got")
	 	if(iteration_count%20>=ITERATION_REPLACE):
	 		MODEL.append(self.model.get_weights())
	 	tf.saved_model.save(self.model,".")
	 def save(self):
	 	self.model.save_weights("weights")
	 def load(self):
	 	self.model.load_weights("weights")

agent=Agent(0.99,(256,240,1))
agent.load()
agent.play()
agent.save()