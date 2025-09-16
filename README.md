# silmarel

## How to install 

- Clone this repository 
- At the beginning of a code/notebook you wish to use this, insert
```
import sys
sys.path.insert(1, 'PATH-TO-REPO')
```
- Import as you would any other package

## Dependencies

Please make sure that the following packages are also installed in any environment used: 

pip-installable: 
- Lenstronomy
- Bilby 
- Jax 
- Arviz
- Astropy
- Optax
- Nifty8

Manual installs: 
- herculens 
- utax

Note: Sometimes a bug occurs where utax gives an indent error. This can be fixed manually in the environment file by removing the extra space the error mentions.