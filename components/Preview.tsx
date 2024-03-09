import React, {useContext, useEffect, useState, useRef} from "react"
import {useHistory} from "react-router-dom"
import {HashLink as Link} from "react-router-hash-link"
import favicon from "../assets/icons/favicon.png"
import {EnableDragContext, MobileContext, SiteHueContext, SiteSaturationContext, SiteLightnessContext, PreviewImageContext,
ImageBrightnessContext, ImageContrastContext, TabContext, ImageInputContext, ImageHueContext, ImageSaturationContext} from "../Context"
import functions from "../structures/Functions"
import "./styles/preview.less"

const Preview: React.FunctionComponent = (props) => {
    const {enableDrag, setEnableDrag} = useContext(EnableDragContext)
    const {mobile, setMobile} = useContext(MobileContext)
    const {siteHue, setSiteHue} = useContext(SiteHueContext)
    const {siteSaturation, setSiteSaturation} = useContext(SiteSaturationContext)
    const {siteLightness, setSiteLightness} = useContext(SiteLightnessContext)
    const {imageBrightness, setImageBrightness} = useContext(ImageBrightnessContext)
    const {imageContrast, setImageContrast} = useContext(ImageContrastContext)
    const {imageHue, setImageHue} = useContext(ImageHueContext)
    const {imageSaturation, setImageSaturation} = useContext(ImageSaturationContext)
    const {previewImage, setPreviewImage} = useContext(PreviewImageContext)
    const {tab, setTab} = useContext(TabContext)
    const {imageInput, setImageInput} = useContext(ImageInputContext)
    const fullscreenRef = useRef(null) as any
    const history = useHistory()

    const getFilter = () => {
        return `hue-rotate(${siteHue - 180}deg) saturate(${siteSaturation}%) brightness(${siteLightness + 50}%)`
    }

    const fullscreen = async (exit?: boolean) => {
        // @ts-ignore
        if (document.fullscreenElement || document.webkitIsFullScreen || exit) {
            try {
                await document.exitFullscreen?.()
                // @ts-ignore
                await document.webkitExitFullscreen?.()
            } catch {
                // ignore
            }
        } else {
            try {
                await fullscreenRef.current?.requestFullscreen?.()
                // @ts-ignore
                await fullscreenRef.current?.webkitRequestFullscreen?.()
            } catch {
                // ignore
            }
        }
    }

    useEffect(() => {
        const handleKey = (event: KeyboardEvent) => {
            if (event.key === " ") {
                fullscreen()
            }
        }
        window.addEventListener("keydown", handleKey, false)
        return () => {
            window.removeEventListener("keydown", handleKey)
        }
    }, [])

    useEffect(() => {
        if (!previewImage) return
        const close = () => {
            setPreviewImage("")
        }
        window.addEventListener("click", close)
        return () => {
            window.removeEventListener("click", close)
        }
    }, [previewImage])

    useEffect(() => {
        if (tab === "watermark") {
            if (previewImage) {
                setImageInput(previewImage)
                setPreviewImage("")
            }
        }
    }, [tab, previewImage])

    if (tab === "watermark") return null
    if (!previewImage) return null

    return (
        <div className="preview" onMouseEnter={() => setEnableDrag(false)}>
            <img ref={fullscreenRef} className="preview-img" src={previewImage} draggable={false} style={{filter: `brightness(${imageBrightness + 100}%) contrast(${imageContrast + 100}%) hue-rotate(${imageHue - 180}deg) saturate(${imageSaturation}%)`}}/>
        </div>
    )
}

export default Preview