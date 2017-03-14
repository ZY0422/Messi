from distutils.core import setup
from distutils.util import convert_path
import os
from fnmatch import fnmatchcase


def find_packages(where='.', exclude=()):
	out = []
	stack = [(convert_path(where), '')]
	while stack:
		where, prefix = stack.pop(0)
		for name in os.listdir(where):
			fn = os.path.join(where, name)
			if ('.' not in name and os.path.isdir(fn) and
				os.path.isfile(os.path.join(fn, '__init__.py'))
			):
				out.append(prefix+name)
				stack.append((fn, prefix+name+'.'))
	for pat in list(exclude) + ['ez_setup', 'distribute_setup']:
		out = [item for item in out if not fnmatchcase(item, pat)]
	return out


setup(name='gpu_ctc',
	version='0.1',
	packages=find_packages(),
	package_data={'': ['*.cu', '*.cuh', '*.c', '*.h', '*.cpp'],
	'gpu_ctc' : ['../build/libwarpctc.so']})