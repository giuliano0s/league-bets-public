# league-bets-public
Public releases of league-bets

## Definition:
League Bets is a pack multi-variate prediction models with target on League of Legends Professional matches results.

#### Current real proof accuracy:
<dl>
  <dt>> Logistic: <b>70%</b></dt>
  <dt>> Binary: <b>NA</b></dt>
  <dt>> Regression: <b>NA (update needed)</b></dt>
</dl>

<hr>

## Current targets are:
<dl>
  <dt>Score</dt>
  <dd>> Winner of the match. 0 for Blue team, 1 for Red team.</dd>
  <dt>totalKills</dt>
  <dd>> Number of total kills in the match.</dd>
</dl>

<hr>

## Major files:
<dl>
  <dt>Predictor_Binary.ipynb</dt>
  <dd>> Trains a range of Classification models to predict <b>Score</b> using 0 or 1</dd>

  <dt>Predictor_Logistic.ipynb</dt>
  <dd>> Trains a range of Logistic models to predict <b>Score</b> using a range of rational numbers between 0 and 1.</dd>

  <dt>Predictor_Regression.ipynb</dt>
  <dd>> Trains a range of regression models to predict <b>totalKills</b> using whole numbers.</dd>

  <dt>testing_model.ipynb</dt>
  <dd>> Manual real proof generation for any model_type, region or season.</dd>

  <dt>ETL_Pipeline.ipynb</dt>
  <dd>> Data scraping pipeline for all regions and seasons.</dd>
</dl>

<hr>

## Utils files:
<dl>
  <dt>scraping_file.py</dt>
  <dd>> Web scraping content. Generate or update data files.</dd>

  <dt>utils_file.py</dt>
  <dd>> Initialize files, define models, data manipulation functions, cache generation and more.</dd>

  <dt>model_file.py</dt>
  <dd>> Final model for validation and real proof.</dd>

  <dt>contants.py</dt>
  <dd>> Constants definition.</dd>
</dl>

<hr>

## Main data files:

<table style="width:100%">
  <tr>
    <th>Name</th>
    <th>Description</th>
    <th>Type</th>
    <th>Class</th>
    <th>Length</th>
  </tr>
  <tr>
    <td>match_list.pkl</td>
    <td>Full match list</td>
    <td>DataFrame</td>
    <td>Treated Data</td>
    <td>≈29000</td>
  </tr>
  <tr>
    <td>match_list_fill.pkl</td>
    <td>Full match list without NaNs</td>
    <td>DataFrame</td>
    <td>Treated Data</td>
    <td>≈29000</td>
  </tr>
  <tr>
    <td>player_data_table.pkl</td>
    <td>Worldwide player info</td>
    <td>DataFrame</td>
    <td>Treated Data</td>
    <td>≈14000</td>
  </tr>
  <tr>
    <td>team_data_table.pkl</td>
    <td>Worldwide team info</td>
    <td>DataFrame</td>
    <td>Treated Data</td>
    <td>≈2300</td>
  </tr>
  <tr>
    <td>regions_cache.json</td>
    <td>Contains "feature_cols" and "train_data" for each season</td>
    <td>Dictionary</td>
    <td>Cache</td>
    <td>3</td>
  </tr>
  <tr>
    <td>feature_cols</td>
    <td>Wich columns to use as features</td>
    <td>Dictionary</td>
    <td>Cache</td>
    <td>Variable by region</td>
  </tr>
  <tr>
    <td>train_data</td>
    <td>Wich regions to use as train data</td>
    <td>Dictionary</td>
    <td>Cache</td>
    <td>Variable by region</td>
  </tr>
  <tr>
    <td>regions_stats_{season}_{model_type}.pkl</td>
    <td>Contains metric results and info about all regions</td>
    <td>DataFrame</td>
    <td>Cache</td>
    <td>Variable by season</td>
  </tr>
  <tr>
    <td>tournaments_{year}.txt</td>
    <td>List of all tournaments of the year</td>
    <td>List</td>
    <td>Raw Data</td>
    <td>Variable by year</td>
  </tr>
</table>

