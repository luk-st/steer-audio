TEST_PROMPTS = [
    # Original 10
    "Upbeat indie pop track with jangly guitars and handclaps, summer road trip vibes",
    "Melancholic jazz ballad with smooth saxophone, walking bassline, late night atmosphere",
    "Energetic drum and bass with synth arpeggios, electronic bleeps, futuristic feel",
    "Acoustic folk song with fingerpicked guitar, gentle harmonies, campfire warmth",
    "Dark synthwave anthem with pulsing bass, retro analog synths, neon city nights",
    "Cheerful bossa nova with nylon string guitar, soft percussion, breezy coastal mood",
    "Heavy metal riff with distorted guitars, thundering drums, aggressive energy",
    "Ambient electronic soundscape with ethereal pads, distant chimes, meditative calm",
    "Funky disco groove with slap bass, wah-wah guitar, rhythmic strings, danceable beat",
    "Minimalist lo-fi hip hop beat with vinyl crackle, mellow Rhodes, jazzy drum loop",
    # 11-20: World Music & Cultural Styles
    "Traditional Irish jig with fiddle, tin whistle, bodhran drums, celtic celebration",
    "Afrobeat groove with polyrhythmic percussion, brass stabs, hypnotic guitar riff",
    "Flamenco piece with passionate acoustic guitar, handclaps, percussive footwork energy",
    "Indian classical fusion with sitar melody, tabla rhythms, meditative drone",
    "Brazilian samba with surdo drums, cavaquinho strumming, carnival energy",
    "Middle Eastern dub with oud melody, electronic beats, desert mysticism",
    "Reggae roots track with offbeat guitar skank, deep bass, one drop drums",
    "Japanese city pop with bright synths, punchy drums, nostalgic 80s shimmer",
    "Balkan brass band with energetic trumpets, tuba bass, wedding celebration chaos",
    "Hawaiian slack key guitar with ukulele, gentle waves, tropical serenity",
    # 21-30: Electronic Subgenres
    "Techno industrial with distorted kicks, metallic textures, relentless machine rhythm",
    "Chillwave with washed-out synths, reverbed vocals, hazy summer nostalgia",
    "Hardstyle anthem with reverse bass, euphoric lead, stadium energy",
    "Glitch hop with chopped samples, wonky bass, playful digital chaos",
    "Progressive house with building arpeggios, lush pads, sunrise festival moment",
    "Dubstep with massive wobble bass, half-time drums, dark underground energy",
    "Vaporwave with slowed samples, reverbed mall music, retro consumerist dreamscape",
    "Breakbeat with chopped drums, funky samples, high energy street dance",
    "Trance anthem with soaring lead, four-on-floor kick, euphoric breakdown",
    "IDM with complex polyrhythms, granular textures, experimental soundscape",
    # 31-40: Rock & Metal Variations
    "90s grunge with fuzzy guitars, angst-filled dynamics, Seattle rain melancholy",
    "Surf rock with reverbed twangy guitar, driving drums, California wave energy",
    "Post-rock crescendo with layered guitars, building dynamics, cinematic emotion",
    "Psychedelic rock with phaser guitars, swirling organs, mind-expanding journey",
    "Punk rock blast with fast power chords, shouted energy, rebellious spirit",
    "Southern rock with slide guitar, boogie rhythm, dusty highway freedom",
    "Shoegaze wall with heavily effected guitars, dreamy vocals, blurred beauty",
    "Progressive rock odyssey with odd time signatures, synth solos, epic storytelling",
    "Stoner rock with downtuned riffs, slow heavy groove, hazy desert vibes",
    "Math rock with intricate tapping, angular rhythms, precise chaos",
    # 41-50: Jazz & Blues Variations
    "Bebop jazz with fast changes, virtuosic piano, swinging drums, smoky club",
    "Delta blues with slide guitar, stomping rhythm, front porch storytelling",
    "Cool jazz with muted trumpet, brushed drums, sophisticated restraint",
    "Chicago electric blues with wailing harmonica, gritty guitar, juke joint energy",
    "Latin jazz with congas, piano montuno, energetic brass, dance floor heat",
    "Free jazz with avant-garde saxophone, chaotic drums, boundary-pushing expression",
    "Smooth jazz with polished production, soft electric piano, easy listening warmth",
    "Gypsy jazz with acoustic guitar runs, violin melody, Parisian café charm",
    "Soul jazz with funky organ, tenor sax, head-nodding groove",
    "Blues rock with overdriven guitar, shuffling drums, roadhouse swagger",
    # 51-60: Classical & Orchestral
    "Baroque chamber music with harpsichord, string quartet, elegant formality",
    "Romantic piano nocturne with expressive dynamics, melancholic beauty, moonlit reverie",
    "Epic orchestral score with full brass, timpani rolls, heroic adventure",
    "Minimalist piano piece with repeating phrases, subtle variations, hypnotic contemplation",
    "String quartet with emotional cello lead, intimate conversation, chamber warmth",
    "Cinematic trailer music with percussion hits, building tension, dramatic reveal",
    "Neoclassical with piano and strings, modern melancholy, elegant simplicity",
    "Choral piece with layered voices, cathedral reverb, sacred transcendence",
    "Film noir score with muted brass, suspenseful strings, shadowy intrigue",
    "Waltz with sweeping strings, graceful three-four time, ballroom elegance",
    # 61-70: Hip Hop & R&B Variations
    "Boom bap with punchy drums, scratched samples, golden era nostalgia",
    "Trap anthem with 808 bass, hi-hat rolls, dark atmospheric pads",
    "Neo-soul with warm keys, silky bassline, intimate late night mood",
    "Old school funk with horn stabs, clavinet riff, party celebration",
    "Cloud rap with dreamy synths, sparse drums, floating atmospheric haze",
    "G-funk with whining synth leads, deep bass, lowrider cruise",
    "Contemporary R&B with layered vocals, minimal production, emotional vulnerability",
    "Drill beat with sliding bass, aggressive hi-hats, street corner intensity",
    "Alternative hip hop with live instruments, jazzy samples, conscious vibes",
    "Phonk with Memphis samples, cowbell, distorted bass, midnight drift",
    # 71-80: Country & Americana
    "Classic country with pedal steel, two-step rhythm, honky tonk heartbreak",
    "Bluegrass breakdown with banjo rolls, fiddle solo, mountain energy",
    "Americana folk rock with mandolin, storytelling guitar, dusty Americana",
    "Outlaw country with gritty vocals, twangy telecaster, rebel spirit",
    "Country pop crossover with bright production, acoustic guitar, radio-ready hook",
    "Western swing with jazzy guitar, fiddle, dancehall Saturday night",
    "Alt-country with melancholic lap steel, understated drums, poetic introspection",
    "Cowboy ballad with lonely harmonica, slow guitar, campfire under stars",
    "Cajun zydeco with accordion, rubboard, Louisiana bayou celebration",
    "Tex-Mex conjunto with accordion, bajo sexto, border town fiesta",
    # 81-90: Pop & Dance Variations
    "80s synth pop with gated reverb drums, bright arpeggios, neon romance",
    "Modern pop with crisp production, catchy hook, radio polish",
    "K-pop inspired with dynamic arrangement, synth drops, energetic choreography feel",
    "Europop with bouncy beat, infectious melody, summer holiday energy",
    "Bedroom pop with intimate production, soft vocals, late night confessional",
    "Dance pop with four-on-floor beat, soaring chorus, club euphoria",
    "Indie electronic with organic textures, synthetic beats, artistic sensibility",
    "Power pop with jangly guitars, big chorus, energetic optimism",
    "Electropop with vocoder vocals, punchy synths, futuristic romance",
    "Tropical house with steel drums, airy drops, beach sunset vibes",
    # 91-100: Unique & Experimental
    "Circus waltz with calliope organ, theatrical drama, whimsical mischief",
    "Space ambient with vast reverbs, cosmic drones, interstellar drift",
    "Steampunk orchestral with mechanical percussion, brass fanfare, Victorian adventure",
    "8-bit chiptune with square wave melody, arpeggio bass, retro game nostalgia",
    "Tribal ambient with ethnic percussion, nature sounds, primal ritual",
    "Gothic darkwave with minor key synths, brooding atmosphere, romantic darkness",
    "Jazz manouche with acoustic guitar, upright bass, swing-era Paris romance",
    "Mariachi with trumpets, guitarrón, romantic serenade, Mexican celebration",
    "Klezmer with clarinet wails, accordion, Eastern European wedding joy",
    "Cinematic ambient with textural layers, emotional swells, contemplative journey",
]

HOLDOUT_PROMPTS = [
    "Bossa-influenced jazz fusion with electric piano voicings, brushed snare, mellow Sunday afternoon",
    "Mongolian throat-singing fused with downtempo beats, hammered dulcimer, vast steppe atmosphere",
    "Bavarian oompah march with tuba, accordion, beer garden conviviality",
    "Sea shanty chorus with stomping feet, accordion drone, salt-stained camaraderie",
    "Modern classical for prepared piano with muted strings, fragile glassy textures, hushed concert hall",
    "Drum'n'bass jungle revival with chopped Amen breaks, ragga vocal stabs, basement rave heat",
    "Spaghetti western theme with twangy reverb guitar, whistling melody, lonely desert showdown",
    "Andean folk with charango, panpipe wind, mountain village procession",
    "Cumbia rebajada with slowed accordion, dusty rhythm box, after-hours street party",
    "Atmospheric black metal with tremolo guitars, blast beats, frostbitten northern wilderness",
    "Mbalax with fast sabar drums, hopscotching guitar, Senegalese dance floor",
    "Neo-folk with cello drone, soft vocal harmonies, candlelit ritual",
    "Modern bedroom jazz with wurlitzer chords, lazy upright bass, study session focus",
    "Hardcore continuum with halfstep dubstep, sub-bass pressure, smoke-filled warehouse",
    "Tuvan folk fused with cinematic strings, igil bow scraping, wide cinematic horizon",
    "New jack swing with crisp drum machine, slap bass, late-80s street romance",
    "Witch house with pitched-down vocals, occult synths, slow trap drums, hazy nightmare",
    "South African amapiano with log drums, jazzy keys, township sunset groove",
    "Country gospel with pedal steel, gentle hammond, choir backing, Sunday morning hope",
    "Liquid drum and bass with rolling breaks, lush Rhodes chords, rainy uplifting drive",
]

NO_LYRICS = ["[inst]"] * len(TEST_PROMPTS)
NO_LYRICS_HOLDOUT = ["[inst]"] * len(HOLDOUT_PROMPTS)

LYRICS_HOLDOUT = [
    """Coffee cooling on the windowsill
    Sunday breeze and time stands still
    Notes float gentle, soft and slow
    Let this Sunday afternoon glow""",
    """Voices rise from frozen ground
    Ancient throats and ancient sound
    Wind across the open plain
    Songs the steppe gives back again""",
    """Glasses raised, the pretzels fly
    Tuba bouncing toward the sky
    Stomp your feet and sing along
    Beer garden night is loud and long""",
    """Heave away and pull the line
    Salt and rope and turpentine
    Brothers shouting at the sea
    Wherever water takes us, free""",
    """Hammers struck on muted wires
    Glassy embers, hidden fires
    Hush descends, the room holds still
    Sound suspended on the sill""",
    """Basement throbbing, low-end deep
    Sleep is something we don't keep
    Breakbeats chopped and ragga calls
    Sweat dripping from the walls""",
    """Tumbleweeds across the square
    Pistols loaded, dusty hair
    Whistle on the desert wind
    No one knows how this will end""",
    """Pipes ascend the mountain air
    Footsteps echo, voices share
    Llamas walk the narrow road
    Carrying a heavy load""",
    """Streetlamps flicker, drum machines
    Slowed-down love and broken means
    Accordion sighs into the night
    Holding on to fading light""",
    """Frozen forests, blackened sky
    Tremolo and battle cry
    Endless winter, endless dawn
    Carry on, carry on""",
    """Sabar pounding faster still
    Sweat and rhythm bend my will
    Dance until the sunrise calls
    Until the city's last drum falls""",
    """Candles burn on wooden floor
    Cello hums beneath the door
    Hands held tight, the circle stays
    Singing through the longest days""",
    """Pages open, lamp burns low
    Wurlitzer keeps the room aglow
    Bassline walks the quiet path
    Numbers, words, the careful math""",
    """Pressure in the lower bass
    Smoke obscures another face
    Half-step pulse beneath the floor
    Crowd asks for nothing more""",
    """Igil bows the open sky
    Strings sustain a long goodbye
    Mountain breath and endless road
    Carrying the ancient code""",
    """Tape decks rolling, drum machines
    Sharp suit dancing, teenage scenes
    Slap bass talk, falsetto cries
    City lights inside her eyes""",
    """Velvet shadows, slowed-down voice
    Every pitch a fragile choice
    Trap drums drag through midnight haze
    Hexes whispered in the maze""",
    """Log drums roll across the yard
    Sunset hits the township hard
    Jazz keys stretch into the night
    Kids dancing in the fading light""",
    """Pedal steel cries in the pew
    Sunday light comes shining through
    Choir lifts the weary soul
    Broken hearts begin to whole""",
    """Rolling breaks beneath the rain
    Rhodes chords ease the city's pain
    Highways glow with passing lights
    Liquid soul through restless nights""",
]

LYRICS = [
    # Original 10
    """Windows down, the map's unfolded wrong
    Radio's playing our favorite song
    Chasing horizons, feeling free
    This is where we're meant to be""",
    """Smoke curls up in amber light
    The saxophone knows my name tonight
    Ice dissolves like promises made
    Another memory starting to fade""",
    """Neon pulse through circuit veins
    Data falling like silver rain
    We are the signal in the noise
    Electric hearts, synthetic joys""",
    """Embers rise to meet the stars
    Passing songs and old guitars
    Pine smoke tangled in your hair
    A little piece of us stays there""",
    """Chrome reflections, rain-slicked streets
    The city's got a heart that never beats
    Analog ghosts in digital skin
    The night machine pulls me in""",
    """Salt air drifting through the door
    Sandy footprints on the floor
    Afternoon moves slow as honey drips
    Sweetness lingering on our lips""",
    """Thunder cracks the hollow sky
    Rise up, no more asking why
    Steel on steel, the sparks ignite
    Warriors of endless night""",
    """Floating in the space between
    Every thought I've ever seen
    Letting go of all I've held
    Into stillness, gently spelled""",
    """Spotlight hits the mirrored ball
    Feel that bassline through the wall
    Everybody moving right
    We're not stopping tonight""",
    """Vinyl crackle, soft and worn
    Three AM and feeling torn
    Drums loop steady, nothing rushed
    In the quiet, peacefully hushed""",
    # 11-20: World Music & Cultural Styles
    """Fiddle sing and feet take flight
    Dancing through the Celtic night
    Raise your glass up to the sky
    Let the old spirits fly""",
    """Rhythm circle never breaks
    Every step the spirit takes
    Moving forward, moving free
    This is how we're meant to be""",
    """Fire burns inside my chest
    These strings know no time for rest
    Passion written in each note
    Every heartbeat stays afloat""",
    """Morning light on temple stone
    Ancient paths I walk alone
    Silence speaks in deeper ways
    Through the mist of endless days""",
    """Colors spinning, drums are loud
    Dancing lost inside the crowd
    Feet won't stop until the dawn
    Celebration carries on""",
    """Sand and stars and endless road
    Carrying this ancient load
    Voices echo through the dunes
    Singing half-forgotten tunes""",
    """Sun goes down on island time
    Everything is feeling fine
    One love keeps us moving on
    Troubles fade before they're gone""",
    """City lights reflect in rain
    Tokyo calls my name again
    Midnight drive on empty streets
    Where the past and future meet""",
    """Trumpets blast and spirits soar
    Dancing harder than before
    Wedding bells and brass so bright
    Celebrating through the night""",
    """Palm trees sway in gentle breeze
    Waves are rolling with such ease
    Island rhythm, island soul
    Here my heart becomes whole""",
    # 21-30: Electronic Subgenres
    """Machines awake beneath the floor
    Industrial beats we can't ignore
    Metal hearts begin to pound
    Lost inside the factory sound""",
    """Summer fading into haze
    Losing track of all my days
    Memories in pastel light
    Drifting gently through the night""",
    """Hands up high, the bass drops hard
    Leave your worries in the yard
    Feel the kick inside your chest
    Tonight we give it all our best""",
    """Pixels scatter, beats collide
    Take this glitchy sonic ride
    Nothing's broken, nothing's wrong
    Chaos makes the perfect song""",
    """Watch the sunrise paint the sky
    Feel the music lift you high
    Every beat a brand new start
    Morning light inside my heart""",
    """Deep below where shadows grow
    Feel the wobble, feel the flow
    Bass so heavy, shakes the ground
    Underground we found our sound""",
    """Shopping malls of yesterday
    Slowly fading, drift away
    Slowed down dreams in pastel hue
    Everything feels strange and new""",
    """Break it down, then build it up
    Can't get enough, can't get enough
    Chopped up beats and funky soul
    Rhythm takes its full control""",
    """Rising higher, touch the stars
    Leave behind these earthly scars
    Melody lifts up my soul
    In this moment, I am whole""",
    """Patterns shift and change their form
    Beautiful the digital storm
    Logic bends but never breaks
    Art from every sound it makes""",
    # 31-40: Rock & Metal Variations
    """Rain falls down on broken dreams
    Nothing's ever what it seems
    Flannel wrapped around my heart
    Falling slowly, falling apart""",
    """Catch the wave before it breaks
    Feel the rush for all our sakes
    Salty air and sunny skies
    Freedom right before my eyes""",
    """Building slowly toward the light
    Darkness fades into the bright
    Every layer tells a tale
    Whispered words upon the gale""",
    """Colors swirl behind my eyes
    Melting into purple skies
    Riding waves of sound and light
    Journey deep into the night""",
    """Three chords fast and nothing more
    This is what we're fighting for
    No pretense, no compromise
    Truth screaming through the lies""",
    """Dust kicks up on the old highway
    Running from yesterday
    Slide guitar and open road
    Carrying this heavy load""",
    """Blurred edges, fading light
    Beautiful distorted sight
    Lost somewhere between the sound
    Floating gently off the ground""",
    """Time signatures bend and break
    Sonic journeys we will take
    Through the cosmos, through the mind
    Leaving normal far behind""",
    """Desert sun beats down so slow
    Heavy riffs and hazy glow
    Fuzz and feedback fill the air
    Drifting without any care""",
    """Counting beats that twist and turn
    Patterns that we have to learn
    Precision wrapped in chaos bright
    Mathematics sounds just right""",
    # 41-50: Jazz & Blues Variations
    """Fingers fly across the keys
    Chasing down these melodies
    Smoke and shadow, notes so fast
    This moment wasn't meant to last""",
    """Sitting on this old porch chair
    Trouble follows everywhere
    Slide guitar knows all my pain
    Storm clouds bringing more of rain""",
    """Soft and cool, the night moves slow
    Trumpet whispers, soft and low
    Elegance in every phrase
    Lost inside this gentle haze""",
    """Harmonica cries out loud
    Playing for the working crowd
    Every note a story told
    Blues deeper than the river's cold""",
    """Congas call and dancers sway
    Music takes the night away
    Heat rises from the floor
    Bodies moving, wanting more""",
    """Notes scatter like startled birds
    Beyond the reach of any words
    Freedom found in broken rules
    Beautiful the sounds of fools""",
    """Easy evening, easy chair
    Saxophone floats through the air
    Nothing rough, nothing wrong
    Just a simple, soothing song""",
    """Swift fingers and a knowing smile
    Playing in the Django style
    Paris streets and café dreams
    Nothing's quite the way it seems""",
    """Organ swells with funky grace
    Keeping up the steady pace
    Head is nodding, soul is fed
    Groove alive until we're dead""",
    """Twelve bars of the honest truth
    Whiskey wisdom, wasted youth
    Guitar screams what words can't say
    Another night blues my way""",
    # 51-60: Classical & Orchestral
    """Harpsichord plays light and clear
    Echoes of another year
    Formal steps and powdered wigs
    Dancing proper, dancing big""",
    """Moon reflects on ivory keys
    Gentle notes float through the trees
    Melancholy, beautiful ache
    Music for the lonely's sake""",
    """Trumpets call the hero's name
    Nothing will ever be the same
    Rising drums announce the day
    Destiny has found its way""",
    """Round and round the pattern goes
    Where it ends nobody knows
    Simple phrases, deep as time
    Meditation, so sublime""",
    """Cellos speak in human voice
    Every bow stroke is a choice
    Intimate and raw and true
    Music bleeding right through you""",
    """Tension builds toward the peak
    Heart is pounding, knees go weak
    Drums explode, the brass takes flight
    Day emerges from the night""",
    """Piano meets the violin
    Where loss ends and hope begins
    Modern tears on classic themes
    Waking gently from old dreams""",
    """Voices rise toward the dome
    Sacred music finding home
    Centuries of prayer and song
    Lifting spirits all day long""",
    """Shadows move behind the blinds
    Mysteries of all kinds
    Brass plays low and strings play tense
    Nothing here makes any sense""",
    """One two three, one two three, spin
    Let the elegant dance begin
    Gowns swirl across the floor
    Graceful as the days of yore""",
    # 61-70: Hip Hop & R&B Variations
    """Drums hit hard from ninety-four
    Golden era back once more
    Scratch the record, bob your head
    Real hip hop is never dead""",
    """808 shakes the parking lot
    Give them everything we got
    Dark and heavy, hard as stone
    Standing in this trap alone""",
    """Candlelight and satin sheets
    Where our two heartbeats meet
    Warm and soft, the music plays
    Lost in love for all our days""",
    """Horns come in, the party starts
    Funky rhythms, happy hearts
    Get up on your feet and move
    Nothing left to prove, just groove""",
    """Floating high above the ground
    Drifting through this cloudy sound
    Nothing solid, nothing real
    Just this weightless way I feel""",
    """Bounce and roll down palm tree streets
    Laid back flows on funky beats
    West coast sun is shining bright
    Everything is gonna be alright""",
    """Whispered words in empty rooms
    Heart still aching, still consumes
    Stripped down, vulnerable, bare
    Showing you I truly care""",
    """Streets speak loud in heavy bass
    Every corner knows my face
    Sliding notes and hard resolve
    Problems that we have to solve""",
    """Live drums and a stand-up bass
    Taking hip hop to new space
    Conscious words above the groove
    Something real we have to prove""",
    """Memphis samples late at night
    Cowbell hits just feel so right
    Drifting sideways down the lane
    Nothing left but this refrain""",
    # 71-80: Country & Americana
    """Steel guitar weeps and cries
    Underneath these Texas skies
    Two-step shuffle, broken heart
    Same old story from the start""",
    """Banjo rolls like mountain streams
    Picking fast my childhood dreams
    Fiddle sawing, feet are stomping
    Appalachian heart is thumping""",
    """Dusty roads and setting sun
    Day is done but we're not done
    Mandolin and stories told
    American dreams of old""",
    """Rules were made for breaking free
    That's the outlaw way to be
    Telecaster, whiskey glass
    Living hard and living fast""",
    """Truck is rolling down the line
    Everything is feeling fine
    Radio on, windows down
    Heading far from this small town""",
    """Fiddle plays a jazzy line
    Saturday and feeling fine
    Swing your partner round the floor
    Dance until we can't no more""",
    """Lap steel moans like autumn wind
    Thinking bout where I have been
    Quiet words and quieter grief
    Finding solace, finding peace""",
    """Campfire smoke and endless stars
    Old harmonica and guitars
    Lonesome cowboy sings his song
    Desert nights are cold and long""",
    """Accordion squeezes out the tune
    Dancing underneath the moon
    Bayou rhythms, zydeco heat
    Shuffling our Cajun feet""",
    """Accordion and bajo sexto play
    Border town on Saturday
    Polka beats and Spanish words
    Sweetest sounds I've ever heard""",
    # 81-90: Pop & Dance Variations
    """Gated snare and neon glow
    Love was fast but time is slow
    Synthesizers fill the night
    Everything is feeling right""",
    """Catchy hook won't leave my head
    Dancing on the words you said
    Polished, perfect, radio bright
    Pop song feelings burning light""",
    """Synchronized we hit the mark
    Dancing lights cut through the dark
    Drop comes hard, the crowd goes wild
    K-pop dreams have got me styled""",
    """Bouncing beat and summer sun
    Holiday has just begun
    Infectious joy in every line
    Europop and feeling fine""",
    """Recording late into the night
    Laptop glow the only light
    Whispered words and soft guitar
    Confessions from where you are""",
    """Four to the floor, we don't stop
    Dancing till the record drops
    Chorus soars above the crowd
    Sing it with me, sing it loud""",
    """Organic meets the synthetic sound
    Art in textures that we've found
    Indie hearts with electric dreams
    Nothing's quite the way it seems""",
    """Big chorus, jangly riff
    Feeling gives my heart a lift
    Optimistic melody
    This is how I want to be""",
    """Vocoder voice from future days
    Love transmitted through the haze
    Synthetic hearts that beat as one
    Tomorrow's already begun""",
    """Steel drum echoes cross the bay
    Sunset ending perfect day
    Tropical drops and summer breeze
    Island living, island ease""",
    # 91-100: Unique & Experimental
    """Step right up, the show's begun
    Mischief here for everyone
    Calliope spins its spell
    Stories only circus tells""",
    """Drifting through the cosmic void
    Emptiness I once avoided
    Stars are singing ancient songs
    Floating where my soul belongs""",
    """Gears are turning, brass is gleaming
    Victorian adventure dreaming
    Steam-powered hearts beat strong
    Clockwork plays the marching song""",
    """Eight-bit hero, pixelated
    High score dreams and dedicated
    Level up and carry on
    Game continues until dawn""",
    """Drums echo through ancient trees
    Spirits dancing on the breeze
    Primal rhythm, primal call
    Nature's music for us all""",
    """Velvet darkness, candle flame
    Whispered secrets, whispered name
    Minor keys and hearts of black
    Down the gothic moonlit track""",
    """Café smoke and Django dreams
    Paris swinging at the seams
    Acoustic strings and upright bass
    Vintage romance, vintage grace""",
    """Trumpets call across the square
    Rose petals floating through the air
    Serenade beneath your window high
    Mariachi love won't ever die""",
    """Clarinet cries tears of joy
    Dancing maiden, dancing boy
    Wedding band plays through the night
    Eastern European delight""",
    """Textures swell and gently fall
    Cinematic, feeling small
    Contemplating all that's been
    Journeying through worlds within""",
]
