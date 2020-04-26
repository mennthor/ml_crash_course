BUILD_DIR = build
ASSET_DIR = assets

TEXC = lualatex
TEXCOPTS = --output-directory=$(BUILD_DIR) --halt-on-error

# Prepare for the tex files being in 'tex' subdir
TEXDIR = tex
export TEXINPUTS=./:./$(TEXDIR):

# Manually specify asset targets, target must be named as script, one target per script
ASSET_TARGETS := $(BUILD_DIR)/$(ASSET_DIR)/01-img-gradient_descent.png
ASSET_TARGETS += $(BUILD_DIR)/$(ASSET_DIR)/01-img-gradient_descent_momentum.png


# Include files should also trigger a rebuild if changed
TEXDEPS = $(TEXDIR)/header.tex $(TEXDIR)/listing_setup.tex $(TEXDIR)/nord_colors.tex

# Define targets = separate PDFs
TARGET01 = 01-Math.Minimizer.Stats


all: $(TARGET01)

$(TARGET01): $(BUILD_DIR)/$(TARGET01).pdf


$(BUILD_DIR)/%.pdf: $(TEXDIR)/%.tex $(TEXDEPS) $(ASSET_TARGETS) | $(BUILD_DIR)
	# $(TEXC) $(TEXCOPTS) $<
	# $(TEXC) $(TEXCOPTS) $<
	latexmk --$(TEXC) -pvc $(TEXCOPTS) $<


# Creates needed assets from script with same name, one img per script
$(BUILD_DIR)/$(ASSET_DIR)/%.png: $(ASSET_DIR)/%.py | $(BUILD_DIR)
	cd $(ASSET_DIR) && python $(subst $(ASSET_DIR)/,,$<)

# Convenience open-doc-targets
open_$(TARGET01):
	open $(BUILD_DIR)/$(TARGET01).pdf

# Directory management
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)
	mkdir -p $(BUILD_DIR)/$(ASSET_DIR)

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean $(TARGET01) open_$(TARGET01)
