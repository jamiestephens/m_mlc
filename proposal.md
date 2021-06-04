## Proposal

This project will aim to create a classification model for determining if a company will go bankrupt.

#### Question/Need

The purpose of this model is to look at over 60 financial variables that point to a company's health and evaluate which have the greatest likelihood of predicting if a company will go bankrupt. 

Both individual and institutional investors would benefit from understanding key statistics or changes in statistics that indicate a higher or lower likelihood that a company will continue operating from one year to the next. Company executives may also benefit from seeing their firm's finances represented in a classification model that indicates bankruptcy is more likely than they may have thought. 

#### Data Description

The data set to be used is labeled 'Polish companies bankruptcy data Data Set' from the UCI Machine Learning Repository, accessible [here](https://archive.ics.uci.edu/ml/datasets/Polish+companies+bankruptcy+data). One individual sample is a single row of data with 64 fields of financial ratios, followed by a 0 or 1 to indicate if the company went bankrupt or not. My expectation is that between 3 and 6 of these financial ratios can be used to create a model that predicts if any given company will go bankrupt. 

#### Tools

I intend to use Python and several Python libraries (including sklearn, matplotlib, and seaborn), and potentially a SQL server to store the approximately 10,000 rows of data.

#### MVP

An MVP for this project would be a selection of pairplots evaluating the metrics that are most relevant to determining a company's bankruptcy, along with a writeup that describes the preliminary findings and next steps.
