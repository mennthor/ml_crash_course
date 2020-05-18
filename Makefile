BUILD_DIR = build
ASSET_DIR = assets

TEXC = lualatex
TEXCOPTS = --output-directory=$(BUILD_DIR) --halt-on-error

# Prepare for the tex files being in 'tex' subdir
TEXDIR = tex
export TEXINPUTS=./:./$(TEXDIR):

# Each script with 0<chapter>-*.py generates one final asset
_ASSET_TARGETS_01 := $(patsubst %.py,%.png,$(wildcard $(ASSET_DIR)/01-*.py))
ASSET_TARGETS_01 = $(subst $(ASSET_DIR),$(BUILD_DIR)/$(ASSET_DIR),$(_ASSET_TARGETS_01))
_ASSET_TARGETS_02 := $(patsubst %.py,%.png,$(wildcard $(ASSET_DIR)/02-*.py))
ASSET_TARGETS_02 = $(subst $(ASSET_DIR),$(BUILD_DIR)/$(ASSET_DIR),$(_ASSET_TARGETS_02))

# Include files should also trigger a rebuild if changed
TEXDEPS = $(TEXDIR)/header.tex $(TEXDIR)/listing_setup.tex $(TEXDIR)/nord_colors.tex

# Define targets = separate PDFs
TARGET01 = 01-Math.Minimizer.Stats
TARGET02 = 02-ML.Basics


all: $(TARGET01) $(TARGET02)

$(TARGET01): $(ASSET_TARGETS_01) $(BUILD_DIR)/$(TARGET01).pdf

$(TARGET02): $(ASSET_TARGETS_02) $(BUILD_DIR)/$(TARGET02).pdf


$(BUILD_DIR)/%.pdf: $(TEXDIR)/%.tex $(TEXDEPS) | $(BUILD_DIR)
	$(TEXC) $(TEXCOPTS) $<
	$(TEXC) $(TEXCOPTS) $<
	# latexmk --$(TEXC) -pvc $(TEXCOPTS) $<


# Creates needed assets from script with same name, one img per script
$(BUILD_DIR)/$(ASSET_DIR)/%.png: $(ASSET_DIR)/%.py | $(BUILD_DIR)
	cd $(ASSET_DIR) && python $(subst $(ASSET_DIR)/,,$<)

# Convenience open-doc-targets
open_$(TARGET01):
	open $(BUILD_DIR)/$(TARGET01).pdf

open_$(TARGET02):
	open $(BUILD_DIR)/$(TARGET02).pdf

# Directory management
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)
	mkdir -p $(BUILD_DIR)/$(ASSET_DIR)

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean $(TARGET01) open_$(TARGET01) $(TARGET02) open_$(TARGET02)
