# Glimpse - Personal Blog

A minimalist Jekyll blog powered by the Lagrange theme, featuring posts on technology, machine learning, algorithms, and personal reflections.

## 🌐 Live Site

- **Production**: [https://unsadkit.me](https://unsadkit.me)
- **Local Development**: [http://localhost:4000](http://localhost:4000)

## 📋 Table of Contents

- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Writing Posts](#writing-posts)
- [Customization](#customization)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)

## Requirements

- **Ruby**: 2.6+ (currently using 2.6.10)
- **Bundler**: 1.17+ (for dependency management)
- **Jekyll**: 3.10.0 (static site generator)

## Quick Start

### 1. Clone and Setup

```bash
# If cloning for the first time
git clone <repository-url>
cd blog

# Install dependencies
bundle install --path vendor/bundle
```

### 2. Run Locally

```bash
# Start the development server
bundle exec jekyll serve

# Your blog will be available at:
# http://localhost:4000
```

### 3. Stop the Server

Press `Ctrl+C` in the terminal where Jekyll is running.

## Project Structure

```
├── _config.yml          # Main Jekyll configuration
├── _data/
│   └── settings.yml     # Site settings (navigation, social links)
├── _includes/           # Reusable HTML components
│   ├── header.html
│   ├── footer.html
│   ├── head.html
│   └── ...
├── _layouts/            # Page templates
│   ├── default.html
│   ├── post.html
│   ├── home.html
│   └── ...
├── _posts/              # Blog posts (Markdown files)
├── _sass/               # SCSS stylesheets
├── _site/               # Generated site (don't edit)
├── assets/              # Static assets (CSS, images)
├── menu/                # Static pages
│   ├── about.md
│   └── archive.md
├── Gemfile              # Ruby dependencies
├── lagrange.gemspec     # Theme gemspec
└── vendor/              # Local gems (don't commit)
```

## Configuration

### Site Settings (`_config.yml`)

```yaml
title: 'Glimpse'
description: 'PoW'
author: 'Samkit Shah'
url: "https://unsadkit.me"
```

### Navigation & Social Links (`_data/settings.yml`)

- **Navigation**: About, Archive
- **Social Links**: GitHub, Twitter, LinkedIn, Email
- **Analytics**: Google Analytics enabled

### Theme Features

- **Pagination**: 5 posts per page
- **SEO**: Optimized with jekyll-seo-tag
- **Feed**: Atom feed generation
- **Sitemap**: Automatic sitemap generation
- **Syntax Highlighting**: Rouge highlighter with kramdown

## Writing Posts

### Post Format

Create new posts in `_posts/` using the format: `YYYY-MM-DD-title.md`

```yaml
---
layout: post
title: "Your Post Title"
author: "Samkit Shah"
categories: journal
tags: [tag1, tag2]
image: assets/img/your-image.jpg  # Optional
---

Your post content in Markdown...
```

### Current Post Categories

Based on existing posts, common categories include:
- **Technology**: Machine Learning, AI, Algorithms
- **Tutorials**: Technical explanations (CLIP, GANs, VAEs, NeRFs)
- **Programming**: Competitive programming solutions
- **Business**: B2B SaaS insights
- **Personal**: Daily reflections

### Adding Images

1. Place images in `assets/img/`
2. Reference in posts: `![Alt text]({{ site.baseurl }}/assets/img/your-image.jpg)`
3. Set as post header image: `image: assets/img/your-image.jpg` in frontmatter

## Customization

### Changing Site Information

1. **Basic Info**: Edit `_config.yml`
2. **Navigation**: Modify `_data/settings.yml`
3. **Social Links**: Update social section in `_data/settings.yml`
4. **Google Analytics**: Change `google-ID` in `_data/settings.yml`

### Styling

- **SCSS Files**: Edit files in `_sass/`
- **Main Styles**: `assets/css/main.css`
- **Syntax Highlighting**: `assets/css/syntax.css`

### Adding New Pages

1. Create `.md` file in `menu/` directory
2. Add frontmatter with `layout: page`
3. Add to navigation in `_data/settings.yml`

## Deployment

### GitHub Pages

1. Push to GitHub repository
2. Enable GitHub Pages in repository settings
3. Set source to main branch

### Custom Domain

1. Add `CNAME` file with your domain
2. Configure DNS records at your domain provider

### Manual Deployment

```bash
# Build the site
bundle exec jekyll build

# Upload _site/ contents to your web server
```

## Development Commands

```bash
# Install dependencies
bundle install --path vendor/bundle

# Serve locally with live reload
bundle exec jekyll serve

# Serve on specific port
bundle exec jekyll serve --port 3000

# Build for production
bundle exec jekyll build

# Build and watch for changes
bundle exec jekyll build --watch

# Serve with drafts included
bundle exec jekyll serve --drafts

# Clean generated files
bundle exec jekyll clean
```

## Troubleshooting

### Common Issues

#### Ruby Version Compatibility

If you encounter gem compatibility issues:

1. **Check Ruby version**: `ruby --version`
2. **Use Ruby 2.6+**: The blog is configured for Ruby 2.6.10+
3. **Install gems locally**: `bundle install --path vendor/bundle`

#### Jekyll Won't Start

```bash
# Clean and reinstall
rm -rf vendor/bundle Gemfile.lock
bundle install --path vendor/bundle
```

#### Sass/SCSS Issues

The blog uses `jekyll-sass-converter 1.5` for Ruby 2.6 compatibility. If you upgrade Ruby, you can update to newer versions.

#### Port Already in Use

```bash
# Find process using port 4000
lsof -i :4000

# Kill the process
kill -9 <PID>

# Or use different port
bundle exec jekyll serve --port 3000
```

### Dependency Information

- **Jekyll**: 3.10.0 (compatible with Ruby 2.6)
- **Theme**: Lagrange 4.0.0
- **Markdown**: Kramdown with GFM parser
- **Highlighting**: Rouge
- **Sass**: Legacy sass gem (end-of-life but stable)

## Content Guidelines

### Post Topics

Current blog covers:
- **Machine Learning**: Deep learning architectures, training techniques
- **Computer Science**: Algorithms, data structures, competitive programming
- **Technology**: Industry insights, B2B SaaS
- **Personal**: Reflections and journal entries

### Writing Style

- Technical posts with clear explanations
- Code snippets with syntax highlighting
- Mathematical notation support via MathJax
- Images and diagrams for complex concepts

## Contributing

### Adding a New Post

1. Create file: `_posts/YYYY-MM-DD-title.md`
2. Add proper frontmatter
3. Write content in Markdown
4. Test locally: `bundle exec jekyll serve`
5. Commit and push changes

### Modifying Theme

1. Edit files in `_sass/`, `_layouts/`, or `_includes/`
2. Test changes locally
3. Commit modifications

## License

This project uses the Lagrange theme under MIT License.

## Author

**Samkit Shah**
- GitHub: [@samkitshah1262](https://github.com/samkitshah1262)
- Twitter: [@samkit1262](https://twitter.com/samkit1262)
- LinkedIn: [samkitshah1262](https://linkedin.com/in/samkitshah1262)
- Email: samkitshah1262@gmail.com

---

*Blog built with ❤️ using Jekyll and the Lagrange theme*
