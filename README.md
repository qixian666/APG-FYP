# Accelerated Proximal Gradient Method

## Project Overview
The Accelerated Proximal Gradient Method (APGM) is an advanced optimization algorithm that combines both proximal gradient methods and acceleration techniques to solve optimization problems more efficiently.

## Objectives
- To develop an efficient implementation of the APGM.
- To apply the APGM to various optimization problems and compare its performance with existing algorithms.

## Features
- Fast convergence rates.
- Flexibility to handle non-smooth functions.
- Comprehensive logging and visualization of optimization progress.

## Installation
To install the necessary dependencies, run:
```bash
pip install -r requirements.txt
```

## Usage
You can use the APGM by invoking the main script:
```bash
python main.py --data <data_file>
```

## Examples
### Example 1
To solve a given optimization problem, use:
```bash
python main.py --data example_data.json
```
### Example 2
For comparative studies with other algorithms, use:
```bash
python main.py --data comparative_data.json --compare
```

## Results
- Detailed results can be found in the `results` directory after running the algorithm.

## References
- Nesterov, Y. (2013). 
- Beck, A. & Teboulle, M. (2009). 

## License
This project is licensed under the MIT License. See the LICENSE file for more information.