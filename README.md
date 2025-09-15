# Learning Neural Antiderivatives

This is the Python reference implementation of our paper
["Learning Neural Antiderivatives"](https://neural-antiderivatives.mpi-inf.mpg.de).

Neural fields offer continuous, learnable representations that extend beyond traditional discrete formats in visual computing. We study the problem of learning neural representations of repeated antiderivatives directly from a function, a continuous analogue of summed-area tables. Although widely used in discrete domains, such cumulative schemes rely on grids, which prevents their applicability in continuous neural contexts. We introduce and analyze a range of neural methods for repeated integration, including both adaptations of prior work and novel designs. Our evaluation spans multiple input dimensionalities and integration orders, assessing both reconstruction quality and performance in downstream tasks such as filtering and rendering. These results enable integrating classical cumulative operators into modern neural systems and offer insights into learning tasks involving differential and integral operators.


Please cite our paper if you refer to our results or use the method or code in your own work:

    @inproceedings{rubab2024antiderivatives,
    title = {Learning Neural Antiderivatives},
    author = {Fizza Rubab and Ntumba Elie Nsampi and Martin Balint and Felix Mujkanovic and
                Hans-Peter Seidel and Tobias Ritschel and Thomas Leimk{\"u}hler},
    booktitle = {Vision, Modeling, and Visualization},
    year = {2025}
    }
