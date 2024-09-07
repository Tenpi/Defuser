import React, {useContext, useEffect, useState, useRef} from "react"
import {useHistory} from "react-router-dom"
import {HashLink as Link} from "react-router-hash-link"
import favicon from "../assets/icons/favicon.png"
import {EnableDragContext, MobileContext, SiteHueContext, SiteSaturationContext, SiteLightnessContext, ThemeContext, 
ImageBrightnessContext, ImageContrastContext, ImageHueContext, ImageSaturationContext, ThemeSelectorContext} from "../Context"
import functions from "../structures/Functions"
import Slider from "react-slider"
import color from "../assets/icons/color.png"
import dark from "../assets/icons/dark.png"
import light from "../assets/icons/light.png"
import "./styles/titlebar.less"

const colorList = {
    "--selection": "rgba(255, 168, 219, 0.302)",
    "--text": "#ff5ba8",
    "--optionsText": "#ff5ba8",
    "--text-alt": "#ff52c0",
    "--buttonBG": "#ff8ab5",
    "--buttonBG2": "#ff84c2",
    "--buttonBGStop": "#ffafe7",
    "--background": "#ffdef7",
    "--titlebarBG": "#ffaad5",
    "--titlebarText": "#ff3ba0",
    "--sidebarBG": "#ffddf7",
    "--footerBG": "#ff6ea8",
    "--sliderBG": "#ff72b6",
    "--sliderButton": "#ff2b92",
    "--inputBG": "#ffa0d0",
    "--inputText": "#000000",
    "--tabSelected": "#ffffff",
    "--progressText": "#000000",
    "--progressBG": "#ffffff",
    "--drop-color": "rgba(226, 26, 143, 0.9)",
    "--controlnetText": "#ff35a4",
    "--controlnetBG": "#ffc3e4",
    "--controlnetButton": "#ff90ca",
    "--settingsButtonText": "#000000"
}

const colorListDark = {
    "--selection": "rgba(88, 16, 58, 0.302)",
    "--text": "#ca0071",
    "--optionsText": "#ca0071",
    "--text-alt": "#ff3cba",
    "--buttonBG": "#780052",
    "--buttonBG2": "#4c0033",
    "--buttonBGStop": "#eb268f",
    "--background": "#170510",
    "--titlebarBG": "#1a0614",
    "--titlebarText": "#bb1a76",
    "--sidebarBG": "#1b0914",
    "--footerBG": "#1b0914",
    "--sliderBG": "#3b031c",
    "--sliderButton": "#83004e",
    "--inputBG": "#1e020e",
    "--inputText": "#e2268c",
    "--tabSelected": "#e2268c",
    "--progressText": "#ffffff",
    "--progressBG": "#000000",
    "--drop-color": "rgba(226, 26, 143, 0.9)",
    "--controlnetText": "#ff35a4",
    "--controlnetBG": "#240814",
    "--controlnetButton": "#37071c",
    "--settingsButtonText": "#000000"
}

const accessibilityColorList = {
    "--selection": "rgba(255, 168, 219, 0.302)",
    "--text": "#000000",
    "--optionsText": "#ffffff",
    "--text-alt": "#ff52c0",
    "--buttonBG": "#ffdeea",
    "--buttonBG2": "#ffe4f2",
    "--buttonBGStop": "#ffafe7",
    "--background": "#ffffff",
    "--titlebarBG": "#ffffff",
    "--titlebarText": "#ffffff",
    "--sidebarBG": "#ffffff",
    "--footerBG": "#ff6ea8",
    "--sliderBG": "#ffe9f9",
    "--sliderButton": "#ffb8da",
    "--inputBG": "#ffeffc",
    "--inputText": "#000000",
    "--tabSelected": "#ffffff",
    "--progressText": "#000000",
    "--progressBG": "#ffe9f9",
    "--drop-color": "rgba(226, 26, 143, 0.9)",
    "--controlnetText": "#ffffff",
    "--controlnetBG": "#fff0f8",
    "--controlnetButton": "#fff0fb",
    "--settingsButtonText": "#000000"
}
   
const accessibilityColorListDark = {
    "--selection": "rgba(255, 168, 219, 0.302)",
    "--text": "#ffffff",
    "--optionsText": "#9b578d",
    "--text-alt": "#ff52c0",
    "--buttonBG": "#6f2c61",
    "--buttonBG2": "#6c254d",
    "--buttonBGStop": "#ffafe7",
    "--background": "#000000",
    "--titlebarBG": "#000000",
    "--titlebarText": "#82336e",
    "--sidebarBG": "#000000",
    "--footerBG": "#ff6ea8",
    "--sliderBG": "#33142a",
    "--sliderButton": "#883162",
    "--inputBG": "#462540",
    "--inputText": "#ffffff",
    "--tabSelected": "#ffffff",
    "--progressText": "#000000",
    "--progressBG": "#33142a",
    "--drop-color": "rgba(226, 26, 143, 0.9)",
    "--controlnetText": "#d088cc",
    "--controlnetBG": "#462540",
    "--controlnetButton": "#441e3a",
    "--settingsButtonText": "#ffffff"
}

interface Props {
    rerender: () => void
}

let pos = 0

const TitleBar: React.FunctionComponent<Props> = (props) => {
    const {enableDrag, setEnableDrag} = useContext(EnableDragContext)
    const {mobile, setMobile} = useContext(MobileContext)
    const {siteHue, setSiteHue} = useContext(SiteHueContext)
    const {siteSaturation, setSiteSaturation} = useContext(SiteSaturationContext)
    const {siteLightness, setSiteLightness} = useContext(SiteLightnessContext)
    const {imageBrightness, setImageBrightness} = useContext(ImageBrightnessContext)
    const {imageContrast, setImageContrast} = useContext(ImageContrastContext)
    const {imageHue, setImageHue} = useContext(ImageHueContext)
    const {imageSaturation, setImageSaturation} = useContext(ImageSaturationContext)
    const [activeDropdown, setActiveDropdown] = useState(false)
    const {theme, setTheme} = useContext(ThemeContext)
    const {themeSelector, setThemeSelector} = useContext(ThemeSelectorContext)
    const ref = useRef<HTMLCanvasElement>(null)
    const history = useHistory()
    const [colorPos, setColorPos] =  useState(0)

    const titleClick = () => {
        history.push("/")
    }

    useEffect(() => {
        const savedHue = localStorage.getItem("siteHue")
        if (savedHue) setSiteHue(Number(savedHue))
        const savedSaturation = localStorage.getItem("siteSaturation")
        if (savedSaturation) setSiteSaturation(Number(savedSaturation))
        const savedLightness = localStorage.getItem("siteLightness")
        if (savedLightness) setSiteLightness(Number(savedLightness))
        const savedTheme = localStorage.getItem("theme")
        if (savedTheme) setTheme(savedTheme)
        const savedThemeSelector = localStorage.getItem("themeSelector")
        if (savedThemeSelector) setThemeSelector(savedThemeSelector)
        const savedImageBrightness = localStorage.getItem("imageBrightness")
        const savedImageContrast = localStorage.getItem("imageContrast")
        const savedImageHue = localStorage.getItem("imageHue")
        const savedImageSaturation = localStorage.getItem("imageSaturation")
    }, [])

    useEffect(() => {
        if (typeof window === "undefined") return
        let colors = theme === "light" ? colorList : colorListDark
        if (themeSelector === "accessibility") colors = theme === "light" ? accessibilityColorList : accessibilityColorListDark
        for (let i = 0; i < Object.keys(colors).length; i++) {
            const key = Object.keys(colors)[i]
            const color = Object.values(colors)[i]
            document.documentElement.style.setProperty(key, functions.rotateColor(color, siteHue, siteSaturation, siteLightness))
        }
        setTimeout(() => {
            props.rerender()
        }, 100)
        localStorage.setItem("siteHue", siteHue)
        localStorage.setItem("siteSaturation", siteSaturation)
        localStorage.setItem("siteLightness", siteLightness)
        localStorage.setItem("theme", theme)
    }, [siteHue, siteSaturation, siteLightness, theme, themeSelector])

    const resetFilters = () => {
        setSiteHue(180)
        setSiteSaturation(100)
        setSiteLightness(50)
        setImageBrightness(0)
        setImageContrast(0)
        setImageHue(180)
        setImageSaturation(100)
        setTimeout(() => {
            props.rerender()
        }, 100)
    }

    const getFilter = () => {
        let key = "--text"
        if (themeSelector === "accessibility") key = "--controlnetButton"
        const bodyStyles = window.getComputedStyle(document.body)
        const color = bodyStyles.getPropertyValue(key)
        return functions.calculateFilter(color)
    }

    const getMarginLeft = () => {
        if (typeof window === "undefined") return "0px"
        let px = window.innerWidth / 2
        let offset = -35
        return `${px + offset}px`
    }

    const changeTheme = () => {
        if (theme === "light") return setTheme("dark")
        if (theme === "dark") return setTheme("light")
    }

    return (
        <div className={`titlebar`} onMouseEnter={() => setEnableDrag(false)}>
            <div className="titlebar-logo-container" onClick={titleClick}>
                <span className="titlebar-hover">
                    <div className="titlebar-text-container">
                        <span className={`titlebar-text`}>Img Diffuse</span>
                    </div>
                </span>
            </div>
            <div className="titlebar-container">
                <div className="titlebar-color-container">
                    <img className="titlebar-color-icon" src={color} style={{filter: getFilter()}} onClick={() => setActiveDropdown((prev) => !prev)}/>
                </div>
                <div className="titlebar-color-container">
                    <img className="titlebar-color-icon" src={theme === "light" ? dark : light} style={{filter: getFilter(), height: "35px", marginLeft: "8px"}} onClick={changeTheme}/>
                </div>
            </div>
            <div className={`title-dropdown ${activeDropdown ? "" : "hide-title-dropdown"}`} style={{left: getMarginLeft()}}>
                <div className="title-dropdown-row">
                    <span className="title-dropdown-text">Hue</span>
                    <Slider className="title-dropdown-slider" trackClassName="title-dropdown-slider-track" thumbClassName="title-dropdown-slider-thumb" onChange={(value) => setSiteHue(value)} min={60} max={300} step={1} value={siteHue}/>
                </div>
                <div className="title-dropdown-row">
                    <span className="title-dropdown-text">Saturation</span>
                    <Slider className="title-dropdown-slider" trackClassName="title-dropdown-slider-track" thumbClassName="title-dropdown-slider-thumb" onChange={(value) => setSiteSaturation(value)} min={30} max={100} step={1} value={siteSaturation}/>
                </div>
                <div className="title-dropdown-row">
                    <span className="title-dropdown-text">Lightness</span>
                    <Slider className="title-dropdown-slider" trackClassName="title-dropdown-slider-track" thumbClassName="title-dropdown-slider-thumb" onChange={(value) => setSiteLightness(value)} min={45} max={75} step={1} value={siteLightness}/>
                </div>
                <div className="title-dropdown-row">
                    <span className="title-dropdown-text">Image Brightness</span>
                    <Slider className="title-dropdown-slider" trackClassName="title-dropdown-slider-track" thumbClassName="title-dropdown-slider-thumb" onChange={(value) => setImageBrightness(value)} min={0} max={50} step={1} value={imageBrightness}/>
                </div>
                <div className="title-dropdown-row">
                    <span className="title-dropdown-text">Image Contrast</span>
                    <Slider className="title-dropdown-slider" trackClassName="title-dropdown-slider-track" thumbClassName="title-dropdown-slider-thumb" onChange={(value) => setImageContrast(value)} min={0} max={50} step={1} value={imageContrast}/>
                </div>
                <div className="title-dropdown-row">
                    <span className="title-dropdown-text">Image Hue</span>
                    <Slider className="title-dropdown-slider" trackClassName="title-dropdown-slider-track" thumbClassName="title-dropdown-slider-thumb" onChange={(value) => setImageHue(value)} min={150} max={210} step={1} value={imageHue}/>
                </div>
                <div className="title-dropdown-row">
                    <span className="title-dropdown-text">Image Saturation</span>
                    <Slider className="title-dropdown-slider" trackClassName="title-dropdown-slider-track" thumbClassName="title-dropdown-slider-thumb" onChange={(value) => setImageSaturation(value)} min={50} max={150} step={1} value={imageSaturation}/>
                </div>
                <div className="title-dropdown-row">
                    <button className="title-dropdown-button" onClick={() => resetFilters()}>Reset</button>
                </div>
            </div>
        </div>
    )
}

export default TitleBar