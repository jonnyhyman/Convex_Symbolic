from setuptools import setup, find_packages

setup(
    name = 'cvx_sym',
    version = '0.0',
    author = 'Jonny Hyman',
    author_email = 'jonnyhyman@gmail.com',
    description = "A canonicalizer and code generator for convex optimization.",
    packages = find_packages(),

    package_data={
        'cvx_sym.templates': ["*.jinja"],

        # Register all parametric functions here
        'cvx_sym.templates.functions.norm2': ["*.jinja"],
    },
    include_package_data=True,

    install_requires = [
        'jinja2',
        'numpy',
    ],

    license='GPLv3',
    url="https://github.com/jonnyhyman/Convex_Symbolic",
    test_suite='nose.collector',
    tests_require=['nose'],
)
