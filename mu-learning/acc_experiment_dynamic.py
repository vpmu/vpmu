#TODO
"""
This expeirment demonstrates mu-learning on the pos'=vel,vel'=acc+p model where p is static.
"""
import random
from decimal import *
getcontext().prec = 3

# region Global constants used in monitors and step function.
A    = Decimal(1)
B    = Decimal(-10)
pmax = Decimal(9)
obsPos = Decimal(90)
T = Decimal(0.1)
FALLBEHIND_DISTANCE = Decimal(100)
MAX_VEL = Decimal(100)
# endregion

#region Model monitors (transcripted by hand; excluding all checks for constants and identities; i.e., T>0 for constant T and pos=pos_post) to enhance readbility.)
def cm(pre_state, post_state, p):
  pos, vel, acc = pre_state
  pos_post, vel_post, acc_post = post_state
  acc_guard   = acc_post == A  and obsPos-(pos+((A+pmax)*(T*T)/2+T*vel)) > ((vel+T*(A+pmax))*(vel+T*(A+pmax)))/(-2*(B+pmax)) 
  brake_guard = acc_post == B
  return acc_guard or brake_guard

def mm(pre_state, acc, post_state, t, p):
  pos, vel, acc_pre = pre_state
  #assert acc_pre == acc,  "expected the pre-state's acceleration value ot be equal to the post-state's acceleration value."
  pos_post, vel_post, acc_post = post_state
  return pos_post == pos + vel*t + (acc+p)*t*t/2 and vel_post == vel + (acc+p)*t
#endregion

#region Definition of the environment.
def done(state):
  pos, vel, acc = state
  if pos >= obsPos: return 1
  if pos > FALLBEHIND_DISTANCE or vel > MAX_VEL: return 2
  else: return 0

def step(state, action, t, p):
  pos, vel, old_acc = state
  pos_post = pos + vel*t + (action+p)*t*t/2
  vel_post  = vel + (action+p)*t
  return (pos_post, vel_post, action)
#endregion

#region Helper methods for mu-learning
def falsify(pre_state, acc, post_state, t, feasible_models):
  def _is_feasible(disturbance_value):
    v = mm(pre_state, acc, post_state, t, disturbance_value)
    return v
  feasible = list(filter(lambda m: _is_feasible(m), feasible_models))
  return feasible

def _available_actions(feasible_models, actions, state):
  def _available_in_all_models(a):
    post_state = (state[0], state[1], a)
    for p in feasible_models:
      a_is_available = cm(state, post_state, p)
      if not a_is_available: return False
    return True
  
  available = list(filter(lambda a: _available_in_all_models(a), actions))
  assert len(available) > 0, "controller was not live."
  return available
#endregion

#region implementation of RL
def initial_agent():
  agent = dict()
  pos = Decimal(0)
  while(pos < FALLBEHIND_DISTANCE + 20):
    vel = Decimal(0)
    while(vel < MAX_VEL + 20):
      agent[(pos, vel,A)] = 0
      agent[(pos, vel, B)] = 0
      vel = vel + Decimal(1)
    pos = pos + Decimal(1)
  return agent

def k(state):
  """the key associated with a state."""
  return (Decimal(int(state[0])), Decimal(int(state[1])), state[2])

def choose(agent, state, actions):
  alpha = 0.7
  if random.randrange(0,100,1)/100.0 > alpha:
    rv = random.choice(actions)
  else:
    rv = exploitative_action(agent, state, actions)
  assert rv in actions
  return rv

def exploitative_action(agent, state, actions):
  best_action = None
  best_score = None
  for a in actions:
    value = agent[k((state[0], state[1], a))]
    if best_action == None or value > best_score:
      best_score = value
      best_action = a
  return best_action

def update(agent, prev_state, action, next_state):
  learnrate = 0.4
  discount = 0.5
  r = Decimal(reward(prev_state, action, next_state))
  prev_key = k((prev_state[0], prev_state[1], action))
  maxd = Decimal(max(map(lambda a: agent[k((next_state[0], next_state[1], a))], [A,B,])))
  prev_value = Decimal(1 - learnrate)*agent[prev_key]
  next_value =  Decimal(learnrate) * (r - maxd)
  agent[prev_key] = prev_value + next_value 
  return agent

def reward(prev_state, action, next_state):
  if obsPos <= next_state[0]:
    return -100
  elif next_state[0] >= FALLBEHIND_DISTANCE or next_state[1] >= MAX_VEL:
    return -10
  else:
    return Decimal(1.0) + (10 / next_state[0])
#endregion


MAX_ITERS     = 1000
MAX_STEPS  		= 100
TIME_STEP    = T
iter_counter = 0
agent        = initial_agent()

while iter_counter < MAX_ITERS:
  #mu-learn:
  print("Iteration %s" % iter_counter)
  prev_state      = None
  curr_state 		= [Decimal(random.randrange(0,30)), Decimal(0), Decimal(B)]
  step_counter    		= 0
  act        		= None
  all_models = [Decimal(1),Decimal(3),Decimal(5),]
  feasible_models = [Decimal(1),Decimal(3),Decimal(5),] # A model is basically just the value of p.
  true_model = random.choice(feasible_models)
  actions         = [A,B,]
  assert not done(curr_state)
  print("Initial state is %s" % curr_state)
  iter_counter = iter_counter + 1
  step_counter = 0
  cumul_reward = 0
  while not done(curr_state) and step_counter < MAX_STEPS:
    if random.randrange(0,100) > 75:
      true_model = random.choice(all_models)
    step_counter += 1
    # compute the feasible models.
    if act is not None and len(feasible_models) > 1:
      assert prev_state is not None and curr_state is not None
      feasible_models = falsify(prev_state, act, curr_state, TIME_STEP, feasible_models)
      if len(feasible_models) == 1:
        print("Found m*=%s" % feasible_models[0])
      else:
        assert len(feasible_models) > 0, "falsified all models!"
        print("\tFeasible models: %s" % feasible_models)
    elif act is not None and len(feasible_models) == 1:
      feasible_models = falsify(prev_state, act, curr_state, TIME_STEP, feasible_models)
      if len(feasible_models) == 0:
        feasible_models = all_models

    # choose the next action and change states.
    available_actions = _available_actions(feasible_models, actions, curr_state)
    print(available_actions)
    act = choose(agent, curr_state, available_actions)
    prev_state = curr_state
    curr_state = step(curr_state, act, TIME_STEP, true_model)
    print("%s --(%s)--> %s" % (prev_state, act, curr_state))
   
    # if we're out of state the go ahead and quit now.
    if not k(curr_state) in agent.keys():
      print("found a terminal state; not updating model.")
      break

    # update the model.
    r = reward(prev_state, act, curr_state)
    cumul_reward = cumul_reward + r
    agent = update(agent, prev_state, act, curr_state)
    #print("New agent is: %s" % agent)
  print("reward was: %s" % cumul_reward)
  assert done(curr_state) != 1, "found crashing state which should not happen."
