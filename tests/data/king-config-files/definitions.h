#define  PHYSICS                 HD
#define  DIMENSIONS              2
#define  COMPONENTS              2
#define  GEOMETRY                SPHERICAL
#define  BODY_FORCE              POTENTIAL
#define  COOLING                 NO
#define  RECONSTRUCTION          LINEAR
#define  TIME_STEPPING           RK2
#define  DIMENSIONAL_SPLITTING   NO
#define  NTRACER                 1
#define  USER_DEF_PARAMETERS     8

/* -- physics dependent declarations -- */

#define  EOS                     IDEAL
#define  ENTROPY_SWITCH          NO
#define  THERMAL_CONDUCTION      NO
#define  VISCOSITY               NO
#define  ROTATING_FRAME          NO

/* -- user-defined parameters (labels) -- */

#define  SIMULATION_TIME         0
#define  MACH                    1
#define  RHO_0                   2
#define  B_EXPONENT              3
#define  R_SCALING               4
#define  THETA                   5
#define  JET_ACTIVE_TIME         6
#define  JET_EPISODES            7

/* [Beg] user-defined constants (do not change this line) */

#define  UNIT_DENSITY            2.40913798411e-27
#define  UNIT_LENGTH             1.7842418e3*CONST_pc
#define  UNIT_VELOCITY           8.8835e7

/* [End] user-defined constants (do not change this line) */

/* -- supplementary constants (user editable) -- */ 

#define  INITIAL_SMOOTHING   NO
#define  WARNING_MESSAGES    NO
#define  PRINT_TO_FILE       YES
#define  INTERNAL_BOUNDARY   NO
#define  SHOCK_FLATTENING    MULTID
#define  CHAR_LIMITING       NO
#define  LIMITER             MINMOD_LIM
