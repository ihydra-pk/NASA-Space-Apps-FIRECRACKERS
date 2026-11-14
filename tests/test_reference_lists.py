"""Reference-list detection on excerpts copied from indexed NASA documents."""

from __future__ import annotations

import pytest

from src.vectorstore.references import asks_for_publications, looks_like_reference_list

# CFI Science Final Report p.12 (publication list).
CFI_PUBLICATIONS = (
    "V., Hicks, M. C., & Williams, F. A. (2012). Can cool flames support quasi-steady alkane droplet burning?. "
    "Combustion and Flame, 159(12), 3583-3588. Nayagam, V., Dietrich, D. L., Hicks, M. C., & Williams, F. A. "
    "(2015). Cool-flame extinction during n-alkane droplet combustion in microgravity. Combustion and Flame, "
    "162(5), 2140-2147. Nayagam, V., Dietrich, D.L. and Williams, F.A., 2017. A Burke-Schumann analysis of "
    "dual-flame structure supported by a burning droplet. International Communications in Heat and Mass "
    "Transfer, 87, pp.84-89. Seshadri, K., Peters, N., Williams, F. A., Nayagam, V., & Paczko, G. (2016). "
    "Asymptotic analysis of quasi- steady n-heptane droplet combustion supported by cool-flame chemistry. "
    "Combustion Theory and Modelling, 20(6), 1118-1130."
)
# "Aspects of Cool-Flame Supported Droplet Combustion" slides, p.18.
CFI_RELATED_PUBLICATIONS = (
    "Related Publications 1. Nayagam, V., Dietrich, D. L., Ferkul, P. V., Hicks, M. C., and Williams, F. A. "
    "(2012). Can cool flames support quasi-steady alkane droplet burning? Combustion and Flame, 159(12), 3583- "
    "3588. 2. Nayagam, V., Dietrich, D.L., Hicks, M.C., and Williams, F.A., “Cool-flame extinction during "
    "n- alkane droplet combustion in microgravity.” Combustion and Flame, Vol. 162, 2015, pp. 2140- 2147."
)
# PSI-117 dissertation p.215 (numbered bibliography).
DISSERTATION_BIBLIOGRAPHY = (
    "194 [9] S.H. Won, B. Jiang, P. Diévart, C.H. Sohn, Y. Ju, Self-sust aining n-heptane cool diffusion "
    "flames activated by ozone, Proceedings of the Combust ion Institute, 35 (2015) 881-888. [10] C.B. Reuter, "
    "S.H. Won, Y. Ju, Flame structure and ignition limit of partially premixed cool flames in a counterflow "
    "burner, Proceedings of the Combust ion Institute, 36 (2017) 1513-1522. [11] C.H. Sohn, H.S. Han, "
    "C.B. Reuter, Y. Ju, S.H. Won, Thermo- kinetic dynamics of near-limit cool diffusion flames, Proceedings "
    "of the Combustion Institute, 36 (2017) 1329- 1337."
)
# CFI Science Final Report p.9: the findings page that answers "what causes
# cool flame extinction". It must never be demoted.
CFI_FINDINGS = (
    "8 Cool flame extinction is controlled primarily by chemical kinetics. As the droplet size decreases, the "
    "residence time in the reaction zone of the gas-phase diffusion flame is reduced to a value corresponding "
    "to which the cool flame becomes statically unstable, and this leads to cool flame extinction. Figure 4 "
    "shows the cool flame extinction diameter as function of ambient pressure for both free-floated and "
    "fiber-supported droplets for pure n-dodecane burning in air."
)
# Farouk slides p.2: a motivation slide that cites papers in passing.
FAROUK_MOTIVATION = (
    "2 Nayagam et al. (CNF 2012) Microgravity droplet experiment Steady cool flame n-C7H16 droplet Atmospheric "
    "pressure Motivation Cool flame has practical relevance to- Multiphase Gas phase Engine knock & multi-stage "
    "auto-ignition [1] Lean flammability limit [2] Spacecraft fire safety [3] Low and NTC kinetics [4] [1] Dec "
    "(PROCI 2009) [2] Reuter et al. (CNF 2016) [3] Nayagam et al. (CNF 2012) [4] Liang & Law (CNF 2017)"
)
# CFI requirements document p.34: prose with an in-text citation.
SRD_PROSE = (
    "droplets in drop towers. Dietrich et al. (2005) used the Japan Microgravity Center 10 s drop tower, a "
    "facility no longer available, and even then could only study a limited parameter space because of the "
    "limited microgravity time. It is quite likely that cool flames were present in some of the experiments."
)


@pytest.mark.parametrize("text", [CFI_PUBLICATIONS, CFI_RELATED_PUBLICATIONS, DISSERTATION_BIBLIOGRAPHY])
def test_reference_lists_are_recognised(text):
    assert looks_like_reference_list(text)


@pytest.mark.parametrize("text", [CFI_FINDINGS, FAROUK_MOTIVATION, SRD_PROSE, ""])
def test_findings_and_prose_with_citations_are_not(text):
    assert not looks_like_reference_list(text)


def test_questions_about_publications_keep_reference_lists():
    assert asks_for_publications("Which papers report cool-flame extinction?")
    assert asks_for_publications("What did Nayagam et al. publish in Combustion and Flame journal?")
    assert not asks_for_publications("What causes cool flame extinction in microgravity?")
