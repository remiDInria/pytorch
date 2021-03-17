#define PY_SSIZE_T_CLEAN
#include <Python.h>

/* Dynamic Programming algorithm which computes the fastest possible
   rematerialization sequence for a given network and memory limit */


/* Helper conversion functions */
long* PySequenceToLongArray(PyObject* pylist) {
  if (!(pylist && PySequence_Check(pylist)))
    return NULL;
  Py_ssize_t len = PySequence_Size(pylist);
  long* result = calloc(len + 1, sizeof(long));
  for(Py_ssize_t i = 0; i < len; ++i) {
    PyObject* item = PySequence_GetItem(pylist, i);
    result[i] = PyLong_AsLong(item);
    Py_DECREF(item);
  }
  result[len] = 0; 
  return result;
}

double* PySequenceToDoubleArray(PyObject* pylist) {
  if (!(pylist && PySequence_Check(pylist)))
    return NULL;
  Py_ssize_t len = PySequence_Size(pylist);
  double* result = calloc(len + 1, sizeof(double));
  for(Py_ssize_t i = 0; i < len; ++i) {
    PyObject* item = PySequence_GetItem(pylist, i);
    result[i] = PyFloat_AsDouble(item);
    Py_DECREF(item); 
  }
  result[len] = 0; 
  return result;
}

long* getLongArray(PyObject* container, const char* attributeName) {
  PyObject* sequence = PyObject_GetAttrString(container, attributeName); 
  long* result = PySequenceToLongArray(sequence);
  Py_DECREF(sequence); 
  return result; 
}

double* getDoubleArray(PyObject* container, const char* attributeName) {
  PyObject* sequence = PyObject_GetAttrString(container, attributeName); 
  double* result = PySequenceToDoubleArray(sequence);
  Py_DECREF(sequence); 
  return result; 
}

/* Main function, with two arguments: a Chain object describing the
   network to be optimized, and the memory limit as an integer */

static PyObject *
persistent_compute_table(PyObject *self, PyObject *args)
{
  PyObject* chain_param;
  int mmax; 
  
  if (!PyArg_ParseTuple(args, "Oi", &chain_param, & mmax))
    return NULL;

  double* fwd_dur = getDoubleArray(chain_param, "forward_durations");
  if (!fwd_dur)
    return NULL;

  double* bwd_dur = getDoubleArray(chain_param, "backward_durations");
  if (!bwd_dur)
    return NULL;

  long* act_sizes = getLongArray(chain_param, "activation_sizes");
  if (!act_sizes)
    return NULL;

  long* act_tot_usages = getLongArray(chain_param, "activation_total_usages");
  if (!act_tot_usages)
    return NULL;

  long* fwd_tmp = getLongArray(chain_param, "forward_memory_tmp_sizes");
  if (!fwd_tmp)
    return NULL;

  long* bwd_tmp = getLongArray(chain_param, "backward_memory_tmp_sizes");
  if (!bwd_tmp)
    return NULL;


  PyObject* chain_length_param = PyObject_GetAttrString(chain_param, "length");
  if (!chain_length_param) return NULL; 
  long chain_length = PyLong_AsLong(chain_length_param);
  Py_DECREF(chain_length_param); 


  /* The opt table stores the duration of the fastest sequence
     OPT(m, i, l) == smallest duration required to compute from
                     layer i to layer l, with memory m

     The what table stores the corresponding choice made to obtain
     this duration: -1 if it is best to start by performing F_i with
     enable_grad() , or a positive value k if it is best to perform
     F_i F_{i+1} ... F_k with no_grad()
  */
  // TODO: Can be optimized by only allocating memory for l >= i
  // TODO: float / int instead of double / long ?
#define OPT(m, i, l) opt[(m)*(chain_length+1)*(chain_length+1) + (i) * (chain_length+1) + (l)]
  double * opt = calloc((mmax+1) * (chain_length+1) * (chain_length+1), sizeof(double));
  if(!opt) {
    return PyErr_NoMemory();
  }

#define WHAT(m, i, l) what[(m)*(chain_length+1)*(chain_length+1) + (i) * (chain_length+1) + (l)]
  long * what = calloc((mmax+1) * (chain_length+1) * (chain_length+1), sizeof(long));
  if(!what) {
    free(opt);
    return PyErr_NoMemory();
  }

  /* Dynamic Programming: Initialization */
  for(long i = 0; i <= chain_length; ++i) {
    long mmin = fmaxl(act_sizes[i] + act_sizes[i+1] + act_tot_usages[i+1] + bwd_tmp[i], 
		      act_sizes[i+1] + act_tot_usages[i+1] + fwd_tmp[i]);
    mmin = fminl(mmin, mmax+1);
    for(long m = 0; m < mmin; ++m)
      OPT(m, i, i) = INFINITY;
    for(long m = mmin; m <= mmax; ++m)
      OPT(m, i, i) = fwd_dur[i] + bwd_dur[i];
  }

  /* Dynamic Programming: Main recursion */
  for(long dist = 1; dist <= chain_length; ++dist)
    for(long i = 0; i <= chain_length - dist; ++i) {
      long l = i + dist;
      long maxCostFWD = 0;
      long mmin = act_sizes[l+1] + act_sizes[i+1] + fwd_tmp[i];
      if (l > i+1) {
	maxCostFWD = fmaxl(maxCostFWD, act_sizes[l-1] + act_sizes[l] + fwd_tmp[l-1]);
	mmin = fmaxl(mmin, act_sizes[l+1] + maxCostFWD);
      }
      mmin = fminl(mmin, mmax+1);
      for(long m = 0; m < mmin; ++m)
	OPT(m, i, l) = INFINITY;
      for(long m = mmin; m <= mmax; ++m) {
	long bestLeaf = -1;
	double sumFw = 0;
	double bestLeafCost = INFINITY;
	for(long j = i+1; j <= l; ++j) {
	  sumFw += fwd_dur[j-1];
	  if (m >= act_sizes[j]) {
	    double cost = sumFw + OPT(m-act_sizes[j], j, l) + OPT(m, i, j-1);
	    if (cost < bestLeafCost) {
	      bestLeafCost = cost;  bestLeaf = j;
	    }
	  }
	}
	double chainCost = INFINITY;
	if (m >= act_tot_usages[i+1])
	  chainCost = OPT(m, i, i) + OPT(m - act_tot_usages[i+1], i+1, l);
	if (bestLeafCost <= chainCost) {
	  OPT(m, i, l) = bestLeafCost;
	  WHAT(m, i, l) = bestLeaf;
	} else {
	  OPT(m, i, l) = chainCost;
	  WHAT(m, i, l) = -1;
	}
      }
    }

  free(fwd_dur); 
  free(bwd_dur); 
  free(act_sizes); 
  free(act_tot_usages); 
  free(fwd_tmp); 
  free(bwd_tmp); 

  PyObject* res_opt = PyList_New(mmax+1); 
  PyObject* res_what = PyList_New(mmax+1); 
  
  /* Convert the result into Python world */
  for(long m = 0; m <= mmax; ++m) {
    PyObject* res_opt_m = PyList_New(chain_length + 1);
    PyList_SET_ITEM(res_opt, m, res_opt_m); 
    PyObject* res_what_m = PyList_New(chain_length + 1);
    PyList_SET_ITEM(res_what, m, res_what_m); 
    for(long i = 0; i <= chain_length; ++i) {
      PyObject* res_opt_m_i = PyDict_New();
      PyList_SET_ITEM(res_opt_m, i, res_opt_m_i);
      PyObject* res_what_m_i = PyDict_New();
      PyList_SET_ITEM(res_what_m, i, res_what_m_i);
      for(long l = i; l <= chain_length; ++l) {
	PyObject* res_l = PyLong_FromLong(l);
	PyObject* res_opt_m_i_l = PyFloat_FromDouble(OPT(m, i, l)); 
	PyDict_SetItem(res_opt_m_i, res_l, res_opt_m_i_l);
	Py_DECREF(res_opt_m_i_l);
	PyObject* res_what_m_i_l;
	long what_m_i_l = WHAT(m, i, l); 
	if(what_m_i_l < 0)
	  res_what_m_i_l = Py_BuildValue("(O)", Py_True);
	else
	  res_what_m_i_l = Py_BuildValue("(Ol)", Py_False, what_m_i_l); 
	PyDict_SetItem(res_what_m_i, res_l, res_what_m_i_l);
	Py_DECREF(res_what_m_i_l); 
	Py_DECREF(res_l);
      }
    }
  }

  free(opt);
  free(what);

  PyObject* result = PyTuple_Pack(2, res_opt, res_what); 
  Py_DECREF(res_opt); 
  Py_DECREF(res_what); 
  return result; 
}



static PyMethodDef dynamic_programs_methods[] = {
    {"persistent_compute_table",  persistent_compute_table, METH_VARARGS,
     "Compute the optimal table with the persistent algorithm."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef dynamic_programs_module = {
    PyModuleDef_HEAD_INIT,
    "dynamic_programs",   /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    dynamic_programs_methods
};

PyMODINIT_FUNC
PyInit_dynamic_programs(void)
{
    return PyModule_Create(&dynamic_programs_module);
}

