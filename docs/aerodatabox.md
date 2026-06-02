# AeroDataBox — verified + implemented

**Status (2026-06-01): verified against the authoritative OpenAPI spec and
implemented.** The live flight collector ([src/live_flights.py](../src/live_flights.py))
now uses AeroDataBox via API.Market instead of FlightAware AeroAPI. This doc
records what was checked and how the claims held up.

## Why we switched

- FlightAware AeroAPI returns airport results in **15-record result sets**,
  and a Personal-tier key is capped at ~10 result sets/minute. ORD has
  ~1,200+ departures in a 48 h window (80+ result sets), so the forward
  schedule alone could take minutes, and the 168 h recent-history pull made
  it worse.
- AeroDataBox's **FIDS "airport departures/arrivals by local time range"**
  endpoint returns *every* departure in a time window in **one request** —
  no 15-record paging. The window is capped at 12 h, so 48 h = 4 requests
  (~4 s) and 168 h = 14 requests.

## Verification (against `openapi-apimarket-v1.json`)

Source of truth: <https://doc.aerodatabox.com/docs/openapi-apimarket-v1.json>
(downloaded and inspected directly).

| Claim in the original sketch | Verified? | Detail |
|---|---|---|
| Base URL `…/aedbx/aerodatabox` | ✅ | `servers[0].url = https://prod.api.market/api/v1/aedbx/aerodatabox` |
| Path `/flights/airports/iata/ORD/{fromLocal}/{toLocal}` | ✅ | spec path `/flights/airports/{codeType}/{code}/{fromLocal}/{toLocal}` (`codeType=iata`, `code=ORD`) |
| GET, "FIDS … by local time range / TIER 2" | ✅ | exact `summary`; `x-badges: [{name: "TIER 2"}]` |
| Window chunked into 12 h pieces | ✅ | `toLocal` doc: "Must be more than beginning of the search range by **no more than 12 hours**." |
| Query params (direction, withLeg, withCancelled, withCodeshared, withCargo, withPrivate) | ✅ | all present (see defaults below) |
| Auth header (sketch said "confirm") | ✅ | securityScheme `apiKey` in header named **`x-magicapi-key`** |
| Time format | ✅ | **local** time, `YYYY-MM-DDTHH:mm` |
| Cheap | ✅ (qualitatively) | Tier 2 = 2 units/request → ~8 units per full 48 h schedule refresh. API.Market lists low-cost plans (Free trial; ~$5/mo entry; PRO 2 $15/mo·24k units; ULTRA 2 $90/mo·240k units). Exact unit allotment of the entry plan varies across sources — confirm in the API.Market dashboard — but a refresh is fractions of a cent either way. |

### Parameter defaults (and what we send)

| param | API default | we send | why |
|---|---|---|---|
| `direction` | `Both` | `Departure` | ORD departures only |
| `withLeg` | `false` | **`true`** | **required**: without it only `movement` is set (departure times + opposite airport) and there is no destination *arrival time*; with it we get both `departure` (ORD) and `arrival` (dest) blocks |
| `withCancelled` | `true` | `true` | keep cancelled/diverted for the disruption label |
| `withCodeshared` | `true` | **`false`** | drop marketing/codeshare duplicates |
| `withCargo` | `true` | **`false`** | passenger ops only (matches BTS) |
| `withPrivate` | `true` | **`false`** | scheduled carriers only |
| `withLocation` | `false` | `false` | no live position needed |

## Response → normalized schema

The 200 body is `{ departures: AirportFlightContract[], arrivals: [...] }`.
For a departure requested with `withLeg=true`, each contract has a
`departure` block (origin = ORD) and an `arrival` block (destination). The
adapter `aerodatabox_to_normalized()` maps:

- `ident` ← `number` (e.g. "UA 2464") or `callSign`
- `reporting_airline` ← `airline.iata` (fallback `airline.icao`)
- `dest` ← `arrival.airport.iata`
- `scheduled_dep_utc` ← `departure.scheduledTime.utc`
- `scheduled_arr_utc` ← `arrival.scheduledTime.utc`
- `cancelled` ← `status ∈ {Canceled, CanceledUncertain}`
- `diverted` ← `status == Diverted`
- `dep_del15` / `arr_del15` ← `revisedTime − scheduledTime ≥ 15 min` for
  completed flights; `NaN` for future schedule rows (no `revisedTime`)

Everything downstream (`build_flight_features` → NWS weather join →
trailing state → `score`) is unchanged: the adapter is the only
provider-specific code, so swapping providers stays a one-function change.

## Caching (to avoid API costs)

1. **Raw-response TTL cache** at `data/live/cache/aerodatabox/`
   (`AERODATABOX_CACHE_TTL_S`, default 600 s) dedupes identical FIDS calls
   within a refresh / across quick re-runs.
2. **Rolling normalized parquet cache**
   (`data/live/flights_recent_cache.parquet`) accumulates recent completed
   flights incrementally (`AERODATABOX_RECENT_CACHE_REFRESH_H`, default 3 h;
   retention `AERODATABOX_RECENT_CACHE_RETENTION_H`, default 168 h) so the
   dashboard never re-pulls the full 168 h history each refresh. Cold-start
   windows the cache doesn't cover fall back to the latest local BTS
   trailing-state snapshot.

Steady-state cost: ~4 schedule chunks + 1 recent chunk ≈ 5 requests
(~10 units) per refresh.

## Auth / config

- Key: `AERODATABOX_KEY` env var, an `aerodatabox.key` file at the repo root
  (git-ignored via `*.key`), or `--api-key`.

### Marketplace selection (API.Market vs RapidAPI)

AeroDataBox is resold through multiple gateways that proxy the **same** API
(identical path `/flights/airports/{codeType}/{code}/{fromLocal}/{toLocal}`,
params, Tier 2, and response schema). The key and the gateway must match.
Select with `AERODATABOX_PROVIDER` (default `apimarket`), `--provider`, or the
dashboard's "Marketplace" selector under the Live source. Verified against
each gateway's OpenAPI spec:

| provider | base URL | auth header(s) |
|---|---|---|
| `apimarket` (default) | `https://prod.api.market/api/v1/aedbx/aerodatabox` | `x-magicapi-key: <key>` |
| `rapidapi` | `https://aerodatabox.p.rapidapi.com` | `X-RapidAPI-Key: <key>` + `X-RapidAPI-Host: aerodatabox.p.rapidapi.com` |

> Note: AeroDataBox's newest plans are marketed as API.Market-exclusive, but
> existing RapidAPI subscriptions still serve the identical endpoint, so a
> RapidAPI key works with `--provider rapidapi`.

## Sources

- AeroDataBox OpenAPI (API.Market): <https://doc.aerodatabox.com/docs/openapi-apimarket-v1.json>
- AeroDataBox docs: <https://doc.aerodatabox.com/>
- API.Market store + plans: <https://api.market/store/aedbx/aerodatabox>
- AeroDataBox pricing: <https://aerodatabox.com/pricing/>
