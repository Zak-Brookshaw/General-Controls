import numpy as np

class Utilities:
    
    def _shape_check(self, ar:np.array, shape:tuple, dtype=None):
        r"""
        This function adds a dimension to vector arrays 

        Parameters
        ----------
        ar : np.array
            DESCRIPTION.
        shape : tuple
            DESCRIPTION.

        Returns
        -------
        None.

        """
        _s = ar.shape
        if _s == shape:
            new_ar = ar
        elif len(_s) == 1 and len(shape) == 2:
            new_ar = ar[:, np.newaxis]
        
        if dtype:
            new_ar = np.array(new_ar, dtype=dtype)
        
        return new_ar