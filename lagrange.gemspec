# frozen_string_literal: true

Gem::Specification.new do |spec|
  spec.name          = "lagrange"
  spec.version       = "4.0.0"
  spec.authors       = ["Paul Le"]
  spec.email         = ["hello@paulle.ca"]

  spec.summary       = "A minimalist Jekyll theme for running a personal blog"
  spec.homepage      = "https://github.com/LeNPaul/Lagrange"
  spec.license       = "MIT"

  spec.files         = `git ls-files -z`.split("\x0").select { |f| f.match(%r!^(assets|_layouts|_includes|_sass|LICENSE|README|CHANGELOG)!i) }

<<<<<<< HEAD
  spec.add_runtime_dependency "jekyll", "~> 4.2"
=======
  spec.add_runtime_dependency "jekyll", "~> 3.9"
>>>>>>> aaa0585 (back again)
  spec.add_runtime_dependency "jekyll-feed", "~> 0.6"
  spec.add_runtime_dependency "jekyll-paginate", "~> 1.1"
  spec.add_runtime_dependency "jekyll-sitemap", "~> 1.3"
  spec.add_runtime_dependency "jekyll-seo-tag", "~> 2.6"
<<<<<<< HEAD
=======
  spec.add_runtime_dependency "jekyll-sass-converter", "~> 1.5"
  spec.add_runtime_dependency "kramdown-parser-gfm", "~> 1.0"
>>>>>>> aaa0585 (back again)

end
