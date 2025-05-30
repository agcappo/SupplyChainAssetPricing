# Supply Chain x Asset Pricing

__A paper that integrates supply chain graphs with risk factor zoo to build asset pricing factors__

The link to our paper draft is [here](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5031617).

Authors: [Agostino Capponi](https://www.columbia.edu/~ac3827/), [Jose Sidaoui](https://ieor.columbia.edu/content/jose-sidaoui-gali), [Jiacheng Zou](https://jiachzou.github.io/). RA support by [Dehan Cui](https://www.linkedin.com/in/dehancui).

Contact Jiacheng [jiachengzou@gmail.com](mailto:jiachengzou@gmail.com) or Jose [jas2545@columbia.edu](mailto:jas2545@columbia.edu) for questions.
## Data

We provide the python code to automate building of 64 firm characteristics from [Freyberger et al, RFS 2020](https://academic.oup.com/rfs/article/33/5/2326/5821383), and these firms' stock returns for users with access to [Wharton Research Data Services (WRDS)](https://wrds-www.wharton.upenn.edu/).  <br>

First, users need to follow the instructions [here](https://docs.google.com/document/d/1hWdw7lofLNZHhWo9tJ_p0OlMmrIHLzIOB44Zy40EPG4/edit?usp=sharing) to manually click a few download links.  <br>

Then, running our python backend code in the jupyter _"FirmFeatures_Calculations.ipynb"_ (located in "Data" folder) produces a _"features.csv"_ file of the panel data of all NYSE, AMEX, and NASDAQ traded stocks in the user-specified date range.  <br>

We document detailed formulas and variable names [here](https://docs.google.com/spreadsheets/d/1L9-sw4nrinA3j_lgsoJaKbawV0w-DLHXKzcfsmd6dGM/edit?usp=sharing).

## Code

_GNN code.ipynb_ is a self-contained jupyter notebook that implements our model in the paper. The first part of the notebook contains all of our functions to implement the benchmark models in our paper (PCA, RP-PCA, NC Ridge, NC LASSO) as well as the full implementation of our Graph-Neural Network model and construction of our asset pricing factors. The notebook may be run in order. The input datasets are "features.csv" produced from the jupyter notebook in the data folder, "ff_mon.csv" which are the monthly FF5 factors, and "features_and_supply_chain.csv" which is our merged features dataset with the supply chain relationship data. 
