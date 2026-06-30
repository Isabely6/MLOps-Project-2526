# Great Expectations quality suite

`expectations/country_quality_suite.json` is a lightweight suite for the
post-target dataset. It checks the country key, positive population, bounded
mortality rate, and country uniqueness. The existing Kedro data-quality nodes
remain the pipeline's enforcement mechanism; this suite can be run in a Great
Expectations checkpoint when that dependency is installed.
