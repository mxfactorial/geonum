this project implements the geometric number spec in math-1-0.md

skip caps when starting sentences

skip periods on single sentence paragraphs

skip apostrophes in words as in "dont" unless it conflicts with another word, "we're" vs "were"

express π as pi in comments and docs

avoid words like "properly", "correctly", "appropriately" and "valid"

say "compute" instead of "calculate"

say "design" instead of "approach"

avoid saying "good" in docs and comments

say "test" or "prove" instead of "validate", "check" or "verify"

avoid the word "should", for example, "perpendicular vectors should have zero dot product" -> "test perpendicular vectors for a zero dot product"

avoid the word "actual"

when fixing unused variable lint errors in tests with underscore prefixes, add "artifact of geonum automation:" comments where a library feature gap is not detected

to get started developing this project, install rust then git clone https://github.com/mxfactorial/geonum

except for `cargo bench` which consumes a lot of time, ask to run test commands listed in README.md tests section before finishing a task