import pickle

filename = 'agents_info'+'.pckl'
f = open(filename, 'rb')
agent_lt, agent_gs = pickle.load(f)
f.close()