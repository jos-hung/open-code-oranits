#!/usr/bin/env python
# Created by "Thieu" at 20:16, 11/06/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from mealpy.evolutionary_based import EP, ES, MA, GA, DE, FPA, CRO, SHADE

from mealpy.swarm_based import ABC, AO, AVOA, ALO, ACOR, AGTO, BA, BFO, BSA, BES, BeesA, COA, CSA, CSO, CoatiOA
from mealpy.swarm_based import DO, DMOA, EHO, ESOA, FA, FFA, FOA, FOX, FFO, GOA, GWO, GJO, GTO, HGS, HHO, HBA, JA
from mealpy.swarm_based import MFO, MPA, MRFO, MSA, MGO, NMRA, NGO, OOA, PSO, PFA, POA
from mealpy.swarm_based import SFO, SHO, SLO, SRSR, SSA, SSO, SSpiderA, SSpiderO, SCSO, SeaHO, STO, ServalOA
from mealpy.swarm_based import TDO, TSO, WOA, ZOA

from mealpy.physics_based import ASO, ArchOA, CDO, EO, EFO, EVO, FLA, HGSO, MVO, NRO, TWO, WDO, RIME
from mealpy.human_based import BRO, BSO, CA, CHIO, FBIO, GSKA, HBO, HCO, ICA, LCO, QSA, SARO, SSDO, SPBO, TLO, TOA, WarSO
from mealpy.bio_based import BBO, BMO, BBOA, EOA, IWO, SBO, SMA, SOA, SOS, TSA, VCS, WHO
from mealpy.system_based import GCO, WCA, AEO
from mealpy.math_based import AOA, CGO, CircleSA, GBO, INFO, PSS, RUN, SCA, SHIO, TS

from mealpy.music_based import HS
from src.models import APO, ARO


dict_optimizer_classes = {
    "APO": APO.OriginalAPO,
    "EP": EP.OriginalEP,
    "ES": ES.OriginalES,
    "MA": MA.OriginalMA,
    "GA": GA.BaseGA,
    "DE": DE.OriginalDE,
    "JADE": DE.JADE,
    "SADE": DE.SADE,
    "SAP-DE": DE.SAP_DE,
    "FPA": FPA.OriginalFPA,
    "CRO": CRO.OriginalCRO,
    "OCRO": CRO.OCRO,
    "SHADE": SHADE.OriginalSHADE,
    "L-SHADE": SHADE.L_SHADE,

    "ABC": ABC.OriginalABC,
    "AO": AO.OriginalAO,

    "ARO": ARO.OriginalARO,
    "CGG-ARO-01": ARO.CGG_ARO_01,     # Proposed algorithm
    "CGG-ARO-02": ARO.CGG_ARO_02,  # Proposed algorithm
    "CGG-ARO-03": ARO.CGG_ARO_03,  # Proposed algorithm

    "AVOA": AVOA.OriginalAVOA,
    "ALO": ALO.OriginalALO,
    "ACOR": ACOR.OriginalACOR,
    "AGTO": AGTO.OriginalAGTO,
    "BA": BA.OriginalBA,
    "BFO": BFO.ABFO,
    "BSA": BSA.OriginalBSA,
    "BES": BES.OriginalBES,
    "BeesA": BeesA.OriginalBeesA,
    "COA": COA.OriginalCOA,
    "CSA": CSA.OriginalCSA,
    "CSO": CSO.OriginalCSO,
    "CoatiOA": CoatiOA.OriginalCoatiOA,
    "DO": DO.OriginalDO,
    "DMOA": DMOA.DevDMOA,
    "EHO": EHO.OriginalEHO,
    "ESOA": ESOA.OriginalESOA,
    "FA": FA.OriginalFA,
    "FFA": FFA.OriginalFFA,
    "FOA": FOA.OriginalFOA,
    "WOA_FOA": FOA.WhaleFOA,
    "FOX": FOX.OriginalFOX,
    "FFO": FFO.OriginalFFO,
    "GOA": GOA.OriginalGOA,
    "GWO": GWO.OriginalGWO,
    "RW-GWO": GWO.RW_GWO,
    "GWO-WOA": GWO.GWO_WOA,
    "GJO": GJO.OriginalGJO,
    "GTO": GTO.OriginalGTO,
    "HGS": HGS.OriginalHGS,
    "HHO": HHO.OriginalHHO,
    "HBA": HBA.OriginalHBA,
    "JA": JA.OriginalJA,
    "L-JA": JA.LevyJA,
    "MFO": MFO.OriginalMFO,
    "MPA": MPA.OriginalMPA,
    "MRFO": MRFO.OriginalMRFO,
    "MSA": MSA.OriginalMSA,
    "MGO": MGO.OriginalMGO,
    "NMRA": NMRA.OriginalNMRA,
    "NGO": NGO.OriginalNGO,
    "OOA": OOA.OriginalOOA,
    "PSO": PSO.OriginalPSO,
    "P-PSO": PSO.P_PSO,
    "C-PSO": PSO.C_PSO,
    "CL-PSO": PSO.CL_PSO,
    "AIW-PSO": PSO.AIW_PSO,
    "LDW-PSO": PSO.LDW_PSO,
    "TVAC-PSO": PSO.HPSO_TVAC,
    "PFA": PFA.OriginalPFA,
    "POA": POA.OriginalPOA,
    "SFO": SFO.OriginalSFO,
    "SHO": SHO.OriginalSHO,
    "SLO": SLO.OriginalSLO,
    "SRSR": SRSR.OriginalSRSR,
    "SSA": SSA.OriginalSSA,
    "SSO": SSO.OriginalSSO,
    "SSpiderA": SSpiderA.OriginalSSpiderA,
    "SSpiderO": SSpiderO.OriginalSSpiderO,
    "SCSO": SCSO.OriginalSCSO,
    "SeaHO": SeaHO.OriginalSeaHO,
    "STO": STO.OriginalSTO,
    "ServalOA": ServalOA.OriginalServalOA,
    "TDO": TDO.OriginalTDO,
    "TSO": TSO.OriginalTSO,
    "WOA": WOA.OriginalWOA,
    "HI-WOA": WOA.HI_WOA,
    "ZOA": ZOA.OriginalZOA,

    "ASO": ASO.OriginalASO,
    "ArchOA": ArchOA.OriginalArchOA,
    "CDO": CDO.OriginalCDO,
    "EO": EO.OriginalEO,
    "M-EO": EO.ModifiedEO,
    "A-EO": EO.AdaptiveEO,
    "EFO": EFO.OriginalEFO,
    "EVO": EVO.OriginalEVO,
    "FLA": FLA.OriginalFLA,
    "HGSO": HGSO.OriginalHGSO,
    "MVO": MVO.OriginalMVO,
    "NRO": NRO.OriginalNRO,
    "TWO": TWO.OriginalTWO,
    "E-TWO": TWO.EnhancedTWO,
    "O-TWO": TWO.OppoTWO,
    "WDO": WDO.OriginalWDO,
    "RIME": RIME.OriginalRIME,

    "BRO": BRO.OriginalBRO,
    "BSO": BSO.OriginalBSO,
    "CA": CA.OriginalCA,
    "CHIO": CHIO.OriginalCHIO,
    "FBIO": FBIO.OriginalFBIO,
    "GSKA": GSKA.OriginalGSKA,
    "HBO": HBO.OriginalHBO,
    "HCO": HCO.OriginalHCO,
    "ICA": ICA.OriginalICA,
    "LCO": LCO.OriginalLCO,
    "QSA": QSA.OriginalQSA,
    "I-QSA": QSA.ImprovedQSA,
    "SARO": SARO.OriginalSARO,
    "SSDO": SSDO.OriginalSSDO,
    "SPBO": SPBO.DevSPBO,
    "TLO": TLO.OriginalTLO,
    "I-TLO": TLO.ImprovedTLO,
    "TOA": TOA.OriginalTOA,
    "WarSO": WarSO.OriginalWarSO,

    "BBO": BBO.OriginalBBO,
    "BMO": BMO.OriginalBMO,
    "BBOA": BBOA.OriginalBBOA,
    "EOA": EOA.OriginalEOA,
    "IWO": IWO.OriginalIWO,
    "SBO": SBO.OriginalSBO,
    "SMA": SMA.OriginalSMA,
    "SOA": SOA.DevSOA,
    "SOS": SOS.OriginalSOS,
    "TSA": TSA.OriginalTSA,
    "VCS": VCS.OriginalVCS,
    "WHO": WHO.OriginalWHO,

    "GCO": GCO.OriginalGCO,
    "WCA": WCA.OriginalWCA,
    "AEO": AEO.OriginalAEO,
    "AAEO": AEO.AugmentedAEO,
    "EAEO": AEO.EnhancedAEO,
    "MAEO": AEO.ModifiedAEO,
    "IAEO": AEO.ImprovedAEO,

    "AOA": AOA.OriginalAOA,
    "CGO": CGO.OriginalCGO,
    "CircleSA": CircleSA.OriginalCircleSA,
    "GBO": GBO.OriginalGBO,
    "INFO": INFO.OriginalINFO,
    "PSS": PSS.OriginalPSS,
    "RUN": RUN.OriginalRUN,
    "SCA": SCA.OriginalSCA,
    "QLE-SCA": SCA.QleSCA,
    "SHIO": SHIO.OriginalSHIO,
    "TS": TS.OriginalTS,

    "HS": HS.OriginalHS
}
