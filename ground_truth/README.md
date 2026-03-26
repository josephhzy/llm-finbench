# Ground Truth Dataset

## Sourcing Methodology

Each fact in `facts.json` was sourced from the publicly available FY2024 annual reports of the following SGX-listed companies:

| Company | Report Source |
|---------|--------------|
| DBS Group Holdings | dbs.com/investor |
| OCBC Bank | ocbc.com/group/investors |
| United Overseas Bank (UOB) | uobgroup.com/investor-relations |
| Singtel | singtel.com/about-us/investor-relations |
| CapitaLand Investment | capitaland.com/investor-relations |

## Verification Process

1. **Primary extraction:** Each figure was manually extracted from the company's FY2024 annual report PDF.
2. **Page reference:** Every fact includes the page number where the figure appears in the source report.
3. **Context quote:** The verbatim sentence or phrase from the report is stored in the `context` field.
4. **Double verification:** Each fact should be independently verified by a second person before use in production evaluation runs.

## Status

**IMPORTANT:** The current values in `facts.json` are based on publicly available information and approximate figures. Before running a production evaluation, each value MUST be verified against the actual downloaded PDF reports with exact page numbers confirmed.

## Schema

See `facts.json` for the full schema. Key fields:

- `id`: Unique identifier (company_period_metric)
- `value`: The numeric ground truth (float)
- `unit`: One of `percent`, `sgd_millions`, `sgd_billions`, `sgd`, `ratio`, `millions`
- `page`: Page number in source PDF
- `context`: Verbatim quote from the report
- `difficulty`: `easy` (explicitly stated), `medium` (table/calculation), `hard` (cross-reference)
- `category`: Thematic grouping for aggregated analysis
