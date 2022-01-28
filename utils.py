import numpy as np
import numpy.ma as ma
from math import log

def reset_weights(m):
	'''
	Try resetting model weights to avoid
	weight leakage.
	'''
	for layer in m.children():
		if hasattr(layer, 'reset_parameters'):
			print(f'Reset trainable parameters of layer = {layer}')
			layer.reset_parameters()

class custom_beam_search_decoder():
	def __init__(self):
		'''self.crop_disease_blinder = {
			'1': ['00'],
			'2': ['00', 'a5'],
			'3': ['b6', 'b8', 'b7', '00', 'b3', 'a9'],
			'4': ['00'],
			'5': ['a7', 'b6', 'b8', '00', 'b7'],
			'6': ['b4', 'a11', '00', 'b5', 'a12']
		}

		self.disease_risk_blinder = {
			'00': ['0'],
			'a11': ['2', '1'],
			'a12': ['2', '1'],
			'a5': ['2'],
			'a7': ['2'],
			'a9': ['2', '3', '1'],
			'b3': ['1'],
			'b4': ['3', '1'],
			'b5': ['1'],
			'b6': ['1'],
			'b7': ['1'],
			'b8': ['1']
		}'''
  
		self.crop_disease_blinder = {
      		0: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
			1: [1, 2, 4, 5, 6, 7, 8, 9, 10, 11],
			2: [1, 2, 3, 4, 7, 8],
			3: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
			4: [1, 2, 3, 5, 6, 7, 8],
			5: [3, 4, 5, 6, 9, 10, 11]
   		}
  
		self.disease_risk_blinder =	{
			0: [1, 2, 3],
			1: [0, 3],
			2: [0, 3],
			3: [0, 1, 3],
			4: [0, 1, 3],
			5: [0],
			6: [0, 2, 3],
			7: [0, 2],
			8: [0, 2, 3],
			9: [0, 2, 3],
			10: [0, 2, 3],
			11: [0, 2, 3]
   		}
  
      
	def decode(self, data, k:list):
		rule = None
		sequences = [[list(), 0.0]]
  
		for state, row in enumerate(data):
			if state > 2:
				raise 

			if state == 1: 
				rule = self.crop_disease_blinder
    
			elif state == 2:
				rule = self.disease_risk_blinder
    
			all_candidates = list()
   
			for i in range(len(sequences)):
				seq, score = sequences[i]
    
				if rule is not None:
					mask = rule[seq[-1]]
					row[mask] = 1e-100
     
				for j in range(len(row)):
					candidate = [seq + [j], score - row[j]]
					all_candidates.append(candidate)
     
			# order all candidates by score
			ordered = sorted(all_candidates, key=lambda tup:tup[1])
			# select k best
			sequences = ordered[:k[state]]  # multi label k best diffrent because of diffrent dimension size
   
		return sequences