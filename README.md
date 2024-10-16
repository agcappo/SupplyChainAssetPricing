# Supply Chain x Asset Pricing

__A project that integrates supply chain graphs with risk factor zoo to build asset pricing factors__

Authors: [Agostino Capponi](https://www.columbia.edu/~ac3827/), [Jose Sidaoui](https://ieor.columbia.edu/content/jose-sidaoui-gali), [Jiacheng Zou](https://jiachzou.github.io/). RA support by [Dehan Cui](https://www.linkedin.com/in/dehancui).

For questions, please contact Jiacheng Zou at _jz3865 [at] columbia.edu_ 




## Data

We provide the python code to automate building of 64 firm characteristics from [Freyberger et al, RFS 2020](https://academic.oup.com/rfs/article/33/5/2326/5821383) for users with access to [Wharton Research Data Services (WRDS)](https://wrds-www.wharton.upenn.edu/).  <br>

First, users need to follow the instructions [here](https://docs.google.com/document/d/1hWdw7lofLNZHhWo9tJ_p0OlMmrIHLzIOB44Zy40EPG4/edit?usp=sharing) to manually click a few download links.  <br>

Then, running our python backend code in the jupyter _"FirmFeatures_Calculations.ipynb"_ (located in "Data" folder) produces a _"features.csv"_ file of the panel data of all NYSE, AMEX, and NASDAQ traded stocks in the user-specified date range.  <br>

We document detailed formulas and variable names [here](https://docs.google.com/spreadsheets/d/1L9-sw4nrinA3j_lgsoJaKbawV0w-DLHXKzcfsmd6dGM/edit?usp=sharing).
