// @ts-check
// Note: type annotations allow type checking and IDEs autocompletion
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

const lightCodeTheme = require('prism-react-renderer').themes.github;
const darkCodeTheme = require('prism-react-renderer').themes.dracula;

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'MatID',
  tagline: 'MatID is a Python package for identifying and analyzing atomistic systems based on their structure.',
  favicon: 'img/favicon.ico',

  stylesheets: [
    {
      href: 'https://cdn.jsdelivr.net/npm/katex@0.13.24/dist/katex.min.css',
      type: 'text/css',
      integrity:
        'sha384-odtC+0UGzzFL/6PNoE8rX/SPcQDXBJ+uRepguP4QkPCm2LBxH3FA3y+fKSiJ+AmM',
      crossorigin: 'anonymous',
    },
  ],

  plugins: [
    [
      '@docusaurus/plugin-ideal-image',
      {
        quality: 70,
        max: 1030, // max resized image's size.
        min: 640, // min resized image's size. if original is lower, use that size.
        steps: 2, // the max number of images generated between min and max (inclusive)
        disableInDev: false,
      },
    ],
  ],

  // Set the production url of your site here
  url: 'https://fairmat-nfdi.github.io',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/matid',
  deploymentBranch: 'gh-pages',
  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'fairmat-nfdi', // Usually your GitHub org/user name.
  projectName: 'matid', // Usually your repo name.

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  // Even if you don't use internalization, you can use this field to set useful
  // metadata like html lang. For example, if your site is Chinese, you may want
  // to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          //routeBasePath: '/', // Serve the docs at the site's root
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
          sidebarPath: require.resolve('./sidebars.js'),
          sidebarCollapsible: false,
        },
        blog: false,
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      }),
    ],
  ],
  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      // Replace with your project's social card
      image: 'img/docusaurus-social-card.jpg',
      navbar: {
        title: '',
        logo: {
          alt: 'MatID logo',
          src: 'img/logo.png',
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'docsSidebar',
            position: 'left',
            label: 'Docs',
          },
          {
            href: 'https://github.com/fairmat-nfdi/matid',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      colorMode: {
        defaultMode: 'light',
        disableSwitch: true,
        respectPrefersColorScheme: false,
      },
      footer: {
        links: [
          {
            title: 'Docs',
            items: [
              {
                label: 'Quick Start',
                to: '/docs/get-started/quick-start',
              },
              {
                label: 'Symmetry-based Clustering',
                to: '/docs/learn/symmetry-based-clustering',
              },
              {
                label: 'Symmetry Analysis',
                to: '/docs/learn/symmetry-analysis',
              },
            ],
          },
          {
            title: 'Links',
            items: [
              {
                label: 'GitHub',
                href: 'https://github.com/fairmat-nfdi/matid',
              },
              {
                label: 'FAIRmat',
                href: 'https://www.fairmat-nfdi.eu/fairmat',
              },
              {
                label: 'NOMAD',
                href: 'https://about.nomad-lab.eu',
              },
            ],
          },
        ],
      },
      prism: {
        theme: lightCodeTheme,
        darkTheme: darkCodeTheme,
      },
    }),
};

module.exports = config;
