# TIStan #

TIStan is a Python 3 package that implements adaptively-annealed thermodynamic integration for model evidence estimation. This package makes use of [PyStan](https://pystan.readthedocs.io/en/latest/)'s implementation of the No U-Turn Sampler for refreshing the sample population at each inverse temperature increment. A paper will soon be published with a detailed description of this package, and it will be linked here at that time.

## Installation ##

This package may be installed using pip:

`pip install TIStan`

TIStan depends on pystan, numpy, and dill. These packages should be installed by pip automatically if they are not present in your Python installation.

## Tests ##

After installation, a dummy test can be run to make sure everything was installed correctly. This test will not produce valid evidence values. From a Python 3 console run,

```
from TIStan.tests import tests
out = tests.dummy_test()
```

To run a more thorough test that should produce valid evidence values for the example problem run,

```
from TIStan.tests import tests
out = tests.full_test()
```
