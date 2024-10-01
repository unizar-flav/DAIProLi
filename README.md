# DAEProLi
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mario-uni/DAIProLi/blob/main/DAIProLi_6_esp.ipynb)

# Overview
DAIProLi is a python script that allows you to analyze the data generated by differential spectroscopy experiments.

# Usage
This software is created in Google Colab. To access it, click on the Google Colab badge above or on this [link](https://colab.research.google.com/github/Mario-uni/DAIProLi/blob/main/DAIProLi_6_esp.ipynb).

**Step 1**: Check the format of the data you have, you need to export your spreadsheet into a csv file. The way the data is saved must be the same as in the following example:
**Example 1**
| Wavelength nm.    | Baseline |  2 | 4 | 6 | ... |
| -------- | ------- | ------- | ------- | ------- | ------- |
| 300    | -0.0002    | 0.0004     | 0.0031     | 0.0034     | ...   |
| 300.5  | -0.0001    | 0.0004     | 0.0029     | 0.0034     | ...   |
| 301    | -0.0002    | 0.0002     | 0.0027     | 0.0032     | ...   |
| ...    | ...   | ...    | ...   | ...   | ..   |

**Step 2**: Load the libraries and functions *(Librerias y funciones)*. This step only needs to be performed once, regardless of the number of datasets processed.

**Step 3**: Run the next cell *(Subir archivo y formato csv)* and do not change the parameters. A message will appear below it asking you to select the file to be uploaded.

**Step 4**: Run *PreprocesaAbsorbance*, in this cell the difference between minimum and maximum in each spectra will be performed.

**Step 5**: In this cell *(Gráfico espectros)* you will be able to plot the spectra recorded for each condition in 2D and 3D. You have the option to adjust all spectra so that their absorbance is zero at a given target wavelength.

**Step 6**: Here, in *Modelo*, the model that may explain the experimental data is written

**Step 7**: In this cell *(Gráfico ΔAbsorbancia vs Volumen (µL))* the data generated from *PreprocesaAbsorbance* will be plotted.

**Step 8**: The **Parámetros** cell is very important. Here you will write the parameters needed by the model and select whether they are fixed or not. If the parameter is fixed, the **procesa** function (responsible for the fitting) will not optimize those values. **Warning:** if the values inputted for the parameters differ greatly from reality even if they are marked as not fixed the fitting process might have worse results or even fail. A good initial estimation, will provide a better fitting.

**Step 9**: Run the *Procesa* cell which will perform the fitting via the least squares method.



**Step 10**: Here *(Gráfico Δ Absorbancia vs Volumen (µL) con el ajuste del modelo)* you can plot the the data genertaed from *PreprocesaAbsorbance* (experimental data) and the values obtained using the model and the optimized parameters from **procesa**.

**Step 11**: Lastly, this cell saves and export all the data inputted and generated. The datasets are saved in csv format and the plots are saved in html and png. All of this items are downloaded in a zip which you can name (but it will always have the date and hour of when the cells was run as suffix).
