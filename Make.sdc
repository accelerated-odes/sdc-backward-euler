AMREX_HOME ?= /path/to/amrex
ODE_SDC_HOME ?= /path/to/here

# default is to compile with CUDA
USE_CUDA ?= FALSE

TOP := $(ODE_SDC_HOME)

EBASE ?= main

# EXTERN_CORE is simply a list of the directories we use for runtime parameters
EXTERN_CORE ?=

#------------------------------------------------------------------------------
# Extra Libraries
#------------------------------------------------------------------------------

# LIBRARIES += -l[library]

#------------------------------------------------------------------------------
# Preprocessor Definitions
#------------------------------------------------------------------------------

# DEFINES += -D[name]

#------------------------------------------------------------------------------
# Standard AMReX Build Definitions
#------------------------------------------------------------------------------

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

all: $(executable)
	@echo SUCCESS

#------------------------------------------------------------------------------
# Directories
#------------------------------------------------------------------------------

Bdirs := Source
Bpack += $(foreach dir, $(Bdirs), $(TOP)/$(dir)/Make.package)
Blocs += $(foreach dir, $(Bdirs), $(TOP)/$(dir))

#------------------------------------------------------------------------------
# AMReX
#------------------------------------------------------------------------------

# core AMReX directories -- note the Make.package for these adds these
# directories into VPATH_LOCATIONS and INCLUDE_LOCATIONS for us, so we
# don't need to do it here

ifeq ($(USE_AMR_CORE), TRUE)
  Pdirs	:= Base AmrCore Amr Boundary
else
  Pdirs := Base
endif

Bpack	+= $(foreach dir, $(Pdirs), $(AMREX_HOME)/Src/$(dir)/Make.package)

Bpack += $(foreach dir, $(EXTERN_CORE), $(dir)/Make.package)
Blocs += $(foreach dir, $(EXTERN_CORE), $(dir))


#------------------------------------------------------------------------------
# include all of the necessary directories
#------------------------------------------------------------------------------

include $(Bpack)

# this is a safety from the mega-fortran attempts
f90EXE_sources += $(ca_f90EXE_sources)
F90EXE_sources += $(ca_F90EXE_sources)

INCLUDE_LOCATIONS += $(Blocs)
VPATH_LOCATIONS   += $(Blocs)

.FORCE:
.PHONE: .FORCE

#------------------------------------------------------------------------------
# finish up
#------------------------------------------------------------------------------

include $(AMREX_HOME)/Tools/GNUMake/Make.rules

# for debugging.  To see the value of a Makefile variable,
# e.g. Fmlocs, simply do "make print-Fmlocs".  This will print out the
# value.

print-%::
	@echo "$* is $($*)"
