import os
import subprocess
import sys
import contextlib

from setuptools import Extension, setup
import numpy


PACKAGES = [
    'forest_cluster',
    'forest_cluster.tests',
]

CYTHON_MODS = [
    'forest_cluster.similarity'
]

@contextlib.contextmanager
def chdir(new_dir):
    old_dir = os.getcwd()
    try:
        sys.path.insert(0, new_dir)
        yield
    finally:
        del sys.path[0]
        os.chdir(old_dir)


def clean(path):
    for name in CYTHON_MODS:
        name = name.replace('.', '/')
        for ext in ['.c', '.cpp', '.so']:
            file_path = os.path.join(path, name + ext)
            if os.path.exists(file_path):
                os.unlink(file_path)


def get_python_package(root):
    return os.path.join(root, 'forest_cluster')


def generate_sources(root):
    for base, _, files in os.walk(root):
        for filename in files:
            if filename.endswith('pyx'):
                yield os.path.join(base, filename)


def generate_cython(root, cython_cov=False):
    print("Cythonizing sources")
    for source in generate_sources(get_python_package(root)):
        cythonize_source(source, cython_cov)


def cythonize_source(source, cython_cov=False):
    print("Processing %s" % source)

    flags = ['--fast-fail']
    if cython_cov:
        flags.extend(['--directive', 'linetrace=True'])
        flags.extend(['--directive', 'binding=True'])

    try:
        p = subprocess.call(['cython'] + flags + [source])
        if p != 0:
            raise Exception('Cython failed')
    except OSError:
        raise OSError('Cython needs to be installed')


def generate_extensions(root, macros=[]):
    ext_modules = []
    for mod_name in CYTHON_MODS:
        mod_path = mod_name.replace('.', '/') + '.c'
        ext_modules.append(
            Extension(mod_name,
                      sources=[mod_path],
                      include_dirs=[numpy.get_include()],
                      extra_compile_args=['-O3', '-fPIC'],
                      define_macros=macros))

    return ext_modules


def setup_package():
    root = os.path.abspath(os.path.dirname(__file__))

    if len(sys.argv) > 1 and sys.argv[1] == 'clean':
        return clean(root)

    cython_cov = 'CYTHON_COV' in os.environ

    macros = []
    if cython_cov:
        print("Adding coverage information to cythonized files.")
        macros =  [('CYTHON_TRACE', 1)]

    with chdir(root):
        generate_cython(root, cython_cov)
        ext_modules = generate_extensions(root, macros=macros)
        setup(
            name="Random Forest Clustering",
            version='0.1.0',
            description='Unsupervised Clustering using Random Forests',
            author='Joshua D. Loyal',
            url='https://github.com/joshloyal/RandomForestClustering',
            license='MIT',
            install_requires=['numpy', 'scipy', 'scikit-learn', 'joblib'],
            packages=PACKAGES,
            ext_modules=ext_modules,
        )


if __name__ == '__main__':
    setup_package()
