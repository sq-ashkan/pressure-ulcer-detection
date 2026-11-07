# Reference and link verification (2026-11-07)

`scripts/verify_references.py docs/thesis/references.bib` checked all 116 entries: DOIs against
Crossref/DataCite (title, year, venue), entries without DOI by their URL. 110 passed automatically.
The six remaining links were opened one by one in a browser on 2026-11-07:

| Key | URL | Result |
|---|---|---|
| pressureinjury2025maskrcnn | universe.roboflow.com/pressure-injury/maskrcnn-swd7w | page loads ("maskrcnn Computer Vision Model") |
| mweber2026dekubitus | universe.roboflow.com/mweber1-rheinahrcampus-de/dekubitus | page loads ("Dekubitus Computer Vision Dataset") |
| eu2017regulation | data.europa.eu/eli/reg/2017/745/oj | redirects to EUR-Lex, Regulation (EU) 2017/745 (MDR) |
| eu2024regulation | data.europa.eu/eli/reg/2024/1689/oj | redirects to EUR-Lex, Regulation (EU) 2024/1689 |
| fid2023pressureulcer | universe.roboflow.com/fid-fvcc6/pressureulcer-ctn4w | Universe page: "Project Not Found"; the Roboflow API reports the project as public, CC BY 4.0 (481 source images, 6 versions); the export used (version 1, 691 images) was downloaded via the API on 2026-11-05. The bibliography note states this. |
| class2023mobile | universe.roboflow.com/class-vx0ys/mobile-application-for-diagnostic-of-pressure-ulcer | same as above: page not found, API reports public, CC BY 4.0, 313 images; note added. |

The script flagged the first four as HTTP 403 (Roboflow blocks scripts) or HTTP 202 (EUR-Lex answers
asynchronously); it now accepts every 2xx status and labels 401/403/429 as "check in a browser".
