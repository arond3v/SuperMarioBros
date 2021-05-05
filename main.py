import tensorflow as tf
from tensorflow.keras.layers import Dense,Flatten,Conv2D,MaxPool2D
from tensorflow.keras.optimizers import Adam
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT,COMPLEX_MOVEMENT
import gym_super_mario_bros
from collections import deque
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
MEMORY_SIZE=100_000
BATCH_SIZE=100
MODEL=[]
iteration_count=0
ITERATION_REPLACE=19
epsilon=0.6
MIN_EPSILON=0.001
EPSILON_DECAY=0.99
target_count=0
TARGET_REPLACE=20
LR=0.001
REP=[]
backup='new'
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
	 	env = JoypadSpace(env,SIMPLE_MOVEMENT)
	 	return env
	 
	 def create_model(self):
	 	model = tf.keras.Sequential()
	 	model.add(Conv2D(64,(3,3),input_shape=self.input_shape))
	 	model.add(MaxPool2D())
	 	model.add(Conv2D(32,(3,3)))
	 	model.add(Flatten())
	 	model.add(Dense(70,activation='relu'))
	 	model.add(Dense(70,activation='relu'))
	 	model.add(Dense(70,activation='relu'))
	 	model.add(Dense(self.action_space,activation='linear'))
	 	model.compile(loss='mse',optimizer=Adam(lr=0.0005),metrics=['accuracy'])
	 	return model

	 def convert(self,state):
	 	img=Image.fromarray(state).convert('L')
	 	return np.array(img.resize((124,120)))
	 def env_step(self,action):
	 	state_=np.zeros(self.input_shape)
	 	reward_=[]
	 	temp=[]
	 	for i in range(2):
	 		self.env.render()
	 		state,reward,done,info=self.env.step(action)
	 		reward_.append(info['x_pos'])
	 		state=self.convert(state)
	 		state_[:,:,i]=state[:,:]
	 	temp=reward_[1]-reward_[0]
	 	if(temp==0):
	 		temp=-3
	 	return state_/255,temp,done,info
	 def env_reset(self):
	 	state_=np.zeros(self.input_shape)
	 	state=self.env.reset()
	 	state=self.convert(state)
	 	for i in range(2):
	 		state_[:,:,i]=state[:,:]
	 	return state_/255
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
	 	temp=np.array(X)
	 	self.model.fit(temp,q_list)
	 	target_count+=1
	 	print(target_count+1)
	 	if(target_count>=TARGET_REPLACE):
	 		self.target.set_weights(self.model.get_weights())
	 		print("\n Target")
	 		target_count=0
	 def save_memory(self):
	 	global epsilon
	 	REWARD=0
	 	random.shuffle(self.memory)
	 	state=self.env_reset()
	 	done=False
	 	flag=False
	 	while(not done):
	 		if(np.random.random()>epsilon):
	 			temp=self.model.predict(state.reshape(1,*self.input_shape))
	 			action=temp[0].argmax()
	 			print("Action",action,end='\r')
	 		else:
	 			action=np.random.randint(0,self.action_space)
	 		prev_state=state
	 		state,reward,done,info=self.env_step(action)
	 		if(info['flag_get']):
	 			reward+=info['time']
	 			flag=True
	 		if(info['time']<(int)(200*epsilon)):
	 			done=True
	 		if(info['life']<2):
	 			done=True
	 			reward-=20
	 		self.memory.append([prev_state,action,reward,state,done])
	 		REWARD+=reward
	 	if(epsilon>MIN_EPSILON):
	 		epsilon*=EPSILON_DECAY
	 	epsilon=max(MIN_EPSILON,epsilon)
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
	 def save(self):
	 	self.model.save_weights(backup)
	 	self.load()
	 	print("Saved")
	 def load(self):
	 	self.model.load_weights(backup)
	 	self.target.set_weights(self.model.get_weights())
	 def twenty(self):
	 	#self.load()
	 	while(True):
	 		self.play(till_flag=False,epoch=20)
	 		self.save()
	 def test(self):
	 	self.env_reset()
	 	for _ in range(100):
	 		self.env_step(6)
agent=Agent(0.9,(120,124,2))
agent.twenty()
