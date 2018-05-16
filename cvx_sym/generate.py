from cvx_sym.operations.functions import Function
from cvx_sym.canonicalize import Canonicalize
from cvx_sym.symbolic import Symbol, Vector
import jinja2
import pathlib

from distutils.dir_util import copy_tree
from zipfile import ZipFile
import urllib.request

import os

class Generate:
    """ Canonicalizes problem and writes the embedded code """

    def __init__(self, problem, name = 'embedded', folder = None,
                                lang = 'c', verbose = False):
        lang = lang.lower()

        self.problem = problem
        self.name = name

        if not folder:
            self.location = pathlib.Path(self.name)
        else:
            self.location = pathlib.Path(folder) / self.name

        self.canonical = Canonicalize(self.problem, verbose = verbose)

        if lang not in ['c']:  # Catch unimplemented languages
            raise(NotImplemented('Only C99 Code Generation Supported'))

        self.write(lang)

    def render(self, name, into):
        """ Activate the template with given name, meaning: write it """

        # Get the template
        template = name + '.jinja'

        # Read the template
        with open(pathlib.Path('cvx_sym/templates') / template, 'r') as f:
            template = f.read()

        # Build the template
        template = jinja2.Template(template)

        # Make directory to copy into
        if not (into).exists():
            into.mkdir()

        # Render and write template
        with open(into / name, 'w') as f:

            render = template.render(self.context)
            f.write(render)

    def pull_solver(self):
        """ Grab solver from __solvers__ folder, and copy into the project """

        # Only copy if doesn't yet exist, to enable faster compilations
        if not (self.location / 'ecos').exists():

            solver_path = pathlib.Path('cvx_sym/__solvers__/ecos')

            copy_tree(str(solver_path.absolute()),
                      str((self.location / 'ecos').absolute()) )

        else:
            print()
            print('NOTE: Did not copy solver because already it exists')

    def adapt_native(self, problem_arrays, lang):
        """ Go through all matrices, looking for parametric functions,
            and represent them in the native language """

        self.native_functions = []
        self.native_func_defs = {}

        for type, arrays in problem_arrays.items():
            for name, array_values in arrays.items():
                for n, value in enumerate(array_values['values']):
                    if hasattr(value, 'parametric') and value.parametric:

                        function = value
                        new_func = ParametricFunction(function, lang)
                        problem_arrays[type][name]['values'][n] = new_func

                        self.native_functions += [new_func]

                        if function.name not in self.native_func_defs:
                            self.native_func_defs[function.name] = new_func

    def write(self, lang):
        """ Activates templates and writes them to disk """

        # Make templates simple by determining most details up in here
        # 'C type' : {'name' : {'values' : [list], 'length' : int }}

        dims  = self.canonical.dims
        vars  = self.canonical.vars
        parms = self.canonical.parms

        problem_arrays = {
            'pfloat' : {
                            'c' : { 'values' : self.canonical.c,
                                    'length' : len(self.canonical.c)},

                            'h' : { 'values' : self.canonical.h,
                                    'length' : len(self.canonical.h)},

                            'Gpr' : { 'values' : self.canonical.G.A,
                                    'length' : len(self.canonical.G.A)},
            },
            'idxint' : {
                            'Gir' : { 'values' : self.canonical.G.JA,
                                    'length' : len(self.canonical.G.JA)},

                            'Gjc' : { 'values' : self.canonical.G.IA,
                                    'length' : len(self.canonical.G.IA)},
            }

        }

        if self.canonical.A is not None:

            problem_arrays['pfloat'].update({

                                'b' : { 'values' : self.canonical.b,
                                        'length' : len(self.canonical.b)},
                                'Apr' : { 'values' : self.canonical.A.A,
                                        'length' : len(self.canonical.A.A)},

            })

            problem_arrays['idxint'].update({

                                'Air' : { 'values' : self.canonical.A.JA,
                                        'length' : len(self.canonical.A.JA)},

                                'Ajc' : { 'values' : self.canonical.A.IA,
                                        'length' : len(self.canonical.A.IA)},
            })

        self.adapt_native(problem_arrays, lang)

        self.context = {}  # Build the template context

        self.context['n'] = len(vars)  # number of vars
        self.context['m'] = len(self.canonical.h)  # number of ineqs

        if self.canonical.A is not None:
            self.context['p'] = len(self.canonical.b)  # number of eqs
        else:
            self.context['p'] = 0 # number of eqs

        self.context['dims'] = dims                      # cone dimensions q, l
        self.context['vars'] = vars                      # variables list
        self.context['parms'] = parms                    # parameters dict
        self.context['do_timing'] = True                 # add timing code?
        self.context['project_name'] = self.name         # name of project
        self.context['problem_arrays'] = problem_arrays  # problem arrays
        self.context['native_functions'] = self.native_functions # add functions
        self.context['native_func_defs'] = self.native_func_defs # add defines

        self.pull_solver()
        self.render('main.c', self.location.absolute())
        self.render('problem.h', self.location.absolute())
        self.render('problem.c', self.location.absolute())
        self.render('CMakeLists.txt', self.location.absolute())

class ParametricFunction:
    """ Object to encapsulate a function which needs to be represented
        natively (since all its arguments are parameters). """

    def __init__(self, func, lang='c'):

        if not issubclass(type(func), Function):
            raise(TypeError(str(type(func))) + " is not a Function subclass. "
                    "ParametricFunction can only accept Functions.")

        functions = pathlib.Path('cvx_sym/templates/functions')

        with open(functions / func.name / 'source.jinja', 'r') as f:
            source = f.read()

        with open(functions / func.name / 'header.jinja', 'r') as f:
            header = f.read()

        with open(functions / func.name / 'caller.jinja', 'r') as f:
            caller = f.read()

        with open(functions / func.name / 'assign.jinja', 'r') as f:
            assign = f.read()

        self.func    = func
        self.args    = func.args
        self.context = {'args' : self.args, 'assign': []}
        self.gather_context()

        self.source = jinja2.Template(source).render()
        self.header = jinja2.Template(header).render()
        self.handle_vectors()

        # Generate string representation
        self.assign = jinja2.Template(assign).render(self.context)
        self.caller = jinja2.Template(caller).render(self.context)

    def handle_vectors(self):
        """ If self.args has Vectors, assign them new variables and
            define those variables via self.runtime """

        for n, arg in enumerate(self.args):
            if type(arg) is Vector:

                sym = Symbol()
                self.context['assign'].append( [sym, arg] )  # sym = arg
                self.args[n] = sym

    def gather_context(self):
        """ Gather all required context from the function itself """
        self.context.update(self.func.get_context())

    def __str__(self):
        return self.caller
