import os
import sys


# path is the directory containing the .xml & .jpg files
# classes is the list of the classes used by the network
def parse_annotation(self, path, classes):

	ann_parsed_file = self.hyperparameters.ann_parsed_file

	if(os.path.isfile(ann_parsed_file)):
		with open(ann_parsed_file,'r') as f:
			dumps = list()
			curdump = list()
			for line in f:
				info = line.strip().split(' ')
				if(info[0]=='#'):
					dumps.append(curdump)
					curdump = list()
					continue
				if(len(info)==1):
					curdump.append(info[0])
					continue
				if(len(info)==2):
					curdump.append([int(info[0]),int(info[1]),[]])
				if(len(info)==5):
					curdump[1][2].append([info[0],int(info[1]),\
						int(info[2]),int(info[3]),int(info[4])])
		return dumps


	#print('Parsing for {} {}'.format(
	#		classes, 'exclusively' * int(exclusive)))
	def pp(l): # pretty printing
		for i in l: print('{}: {}'.format(i,l[i]))

	def parse(line): # exclude the xml tag
		x = line.split('>')[1].split('<')[0]
		try: r = int(x)
		except: r = x
		return r

	def _int(literal): # for literals supposed to be int
		return int(float(literal))

	dumps = list() # the result list
	cur_dir = os.getcwd()
	os.chdir(path)
	annotations = os.listdir('.')
	annotations = [file for file in annotations if '.xml' in file]
	# annotations contains the list of .xml files
	size = len(os.listdir('.'))

	for i, file in enumerate(annotations):

		# progress bar
		sys.stdout.write('\r')
		percentage = 1. * (i+1) / size
		progress = int(percentage * 20)
		bar_arg = [progress*'=', ' '*(19-progress), percentage*100]
		bar_arg += [file]
		sys.stdout.write('[{}>{}]{:.0f}%  {}'.format(*bar_arg))
		sys.stdout.flush()

		# actual parsing
		with open(file, 'r') as f:
			lines = f.readlines()
		w = h = int()
		all = current = list() # current is the current object object
		name = str()
		obj = False
		flag = False
		for i in range(len(lines)):
			line = lines[i]
			if '<filename>' in line:
				jpg = str(parse(line)) # image described in the jpg file
			if '<width>' in line:
				w = _int(parse(line))
			if '<height>' in line:
				h = _int(parse(line)) # dimensions
			if '<object>' in line:
				obj = True
			if '</object>' in line:
				obj = False
			if '<part>' in line:
				obj = False
				# object is composed of several subobjects, but they are not taken into account
			if '</part>' in line:
				obj = True
			if not obj: continue
			if '<name>' in line:
				if current != list():
					if current[0] in classes:
						all += [current] # all is the list of the objects contained in the file
					#elif exclusive: # WHAT IS EXCLUSIVE ?
					#	flag = True
					#	break
				current = list()
				name = str(parse(line))
				if name not in classes:
					obj = False
					continue
				current = [name,None,None,None,None]
			if len(current) != 5: continue
			xn = '<xmin>' in line
			xx = '<xmax>' in line
			yn = '<ymin>' in line
			yx = '<ymax>' in line
			if xn: current[1] = _int(parse(line))
			if xx: current[3] = _int(parse(line))
			if yn: current[2] = _int(parse(line))
			if yx: current[4] = _int(parse(line)) # reads the coordinates of the object

		#if flag: continue
		if current != list() and current[0] in classes:
			all += [current]

		add = [[jpg, [w, h, all]]]
		dumps += add

	# gather all stats in the end
	stat = dict()
	for dump in dumps:
		all = dump[1][2]
		for current in all:
			if current[0] in classes:
				if current[0] in stat:
					stat[current[0]]+=1
				else:
					stat[current[0]] =1

	print()
	print('Statistics:')
	pp(stat)
	print('Dataset size: {}'.format(len(dumps)))

	os.chdir(cur_dir)

	f = open(ann_parsed_file,'w')

	for pack in dumps:

		f.write(pack[0]+'\n')
		f.write(str(pack[1][0])+' '+str(pack[1][1])+'\n')

		for obj in pack[1][2]:
			f.write(obj[0]+' '+str(obj[1])+' '\
				+str(obj[2])+' '+str(obj[3])+' '+str(obj[4])+'\n')

		f.write('#'+'\n')
	return dumps

# ann = "./Annotations"
# classes = ["person", "bottle", "car", "dog", "cat"]
# l = parse_annotation (ann, classes)
#for tab in l :
#	print (tab)
