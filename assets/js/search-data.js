// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-about",
    title: "about",
    section: "Navigation",
    handler: () => {
      window.location.href = "/";
    },
  },{id: "nav-blog",
          title: "blog",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/blog/";
          },
        },{id: "nav-tutorials",
          title: "tutorials",
          description: "Tutorials by Justin",
          section: "Navigation",
          handler: () => {
            window.location.href = "/tutorials/";
          },
        },{id: "nav-teaching",
          title: "teaching",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/teaching/";
          },
        },{id: "nav-cv",
          title: "cv",
          description: "Please click the &quot;PDF&quot; button on this webpage to download my CV.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/assets/pdf/cv_justinBurzachiello.pdf";
          },
        },{id: "post-jax-for-physics-informed-source-separation",
        
          title: "JAX for Physics–Informed Source Separation",
        
        description: "Formulation and numerical solution of a PDE source separation inverse problem that is regularized with physics--informed a priori information.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2025/jax-for-physics-informed-source-separation/";
          
        },
      },{id: "books-the-godfather",
          title: 'The Godfather',
          description: "",
          section: "Books",handler: () => {
              window.location.href = "/books/the_godfather/";
            },},{id: "news-i-recieved-my-m-s-in-computational-applied-mathematics-from-stony-brook-university-in-2024-sbu-was-ranked-8th-in-mathematics-in-the-usa-by-the-academic-ranking-of-world-universities",
          title: 'I recieved my M.S. in Computational Applied Mathematics from Stony Brook University. In...',
          description: "",
          section: "News",},{id: "news-i-started-my-phd-in-mechanical-and-aerospace-engineering-at-the-university-of-california-san-diego-in-2025-u-s-news-and-world-report-ranked-ucsd-10th-among-the-nation-s-top-engineering-schools-and-1-for-citations-per-publication-among-public-universities-in-the-us",
          title: 'I started my PhD in Mechanical and Aerospace Engineering at the University of...',
          description: "",
          section: "News",},{id: "news-i-attended-the-nsf-sponsored-2025-structure-preserving-scientific-computing-and-machine-learning-summer-school-and-hackathon-at-the-university-of-washington-seattle-from-06-16-06-25-while-there-i-studied-dynamical-low-rank-approximations-operator-splitting-and-physics-informed-neural-networks-i-also-helped-implement-a-physics-informed-preconditioner-for-simulating-x-ray-radiative-transport-in-inertial-confiment-fusion",
          title: 'I attended the NSF–sponsored 2025 Structure–Preserving Scientific Computing and Machine Learning Summer School...',
          description: "",
          section: "News",},{id: "news-i-attended-uc-san-diego-s-2025-uc-leads-summer-session-dinner-reception-on-07-01-as-a-recipient-of-the-uc-leads-graduate-fellowship-as-an-undergrad-in-uc-leads-i-was-awarded-the-2022-graduate-deans-leadership-award-for-extraordinary-leadership-funded-by-all-10-university-of-california-graduate-divisions",
          title: 'I attended UC San Diego’s 2025 UC LEADS summer session dinner reception on...',
          description: "",
          section: "News",},{id: "news-i-was-accepted-to-attend-the-nextprof-pathfinder-workshop-at-uc-san-diego-from-10-05-10-07-sponsored-by-the-university-of-michigan-uc-san-diego-and-georgia-tech-this-workshop-prepares-participants-for-a-successful-career-in-academia",
          title: 'I was accepted to attend the NextProf Pathfinder workshop at UC San Diego...',
          description: "",
          section: "News",},{id: "news-i-was-invited-to-present-at-a-uc-leads-workshop-on-08-05-2025-on-how-to-interweave-stories-of-adverse-life-experiences-into-graduate-school-applications-as-a-former-foster-youth-i-find-it-important-to-effectively-convey-triumph-over-adversity-sadly-amp-lt-4-of-foster-youth-obtain-a-4-year-degree-yet-studies-have-shown-that-upwards-of-30-of-foster-care-alumni-suffer-from-ptsd-nearly-more-than-that-of-vietnam-war-veterans",
          title: 'I was invited to present at a UC LEADS workshop on 08/05/2025 on...',
          description: "",
          section: "News",},{id: "projects-project-1",
          title: 'project 1',
          description: "with background image",
          section: "Projects",handler: () => {
              window.location.href = "/projects/1_project/";
            },},{id: "projects-x-ray-computed-tomography",
          title: 'X--Ray Computed Tomography',
          description: "In progress",
          section: "Projects",handler: () => {
              window.location.href = "/projects/ct_tutorial/";
            },},{
        id: 'social-email',
        title: 'email',
        section: 'Socials',
        handler: () => {
          window.open("mailto:%6A%75%62%75%72%7A%61%63%68%69%65%6C%6C%6F@%75%63%73%64.%65%64%75", "_blank");
        },
      },{
        id: 'social-linkedin',
        title: 'LinkedIn',
        section: 'Socials',
        handler: () => {
          window.open("https://www.linkedin.com/in/justin-burzachiello", "_blank");
        },
      },{
      id: 'light-theme',
      title: 'Change theme to light',
      description: 'Change the theme of the site to Light',
      section: 'Theme',
      handler: () => {
        setThemeSetting("light");
      },
    },
    {
      id: 'dark-theme',
      title: 'Change theme to dark',
      description: 'Change the theme of the site to Dark',
      section: 'Theme',
      handler: () => {
        setThemeSetting("dark");
      },
    },
    {
      id: 'system-theme',
      title: 'Use system default theme',
      description: 'Change the theme of the site to System Default',
      section: 'Theme',
      handler: () => {
        setThemeSetting("system");
      },
    },];
